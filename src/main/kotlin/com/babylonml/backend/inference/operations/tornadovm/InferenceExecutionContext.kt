package com.babylonml.backend.inference.operations.tornadovm

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.inference.operations.Operation
import com.babylonml.backend.inference.tornadovm.ContextMemory
import com.babylonml.backend.tornadovm.TvmCommons
import com.babylonml.backend.tornadovm.TvmVectorOperations
import com.babylonml.backend.training.execution.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.ImmutableTaskGraph
import uk.ac.manchester.tornado.api.TaskGraph
import uk.ac.manchester.tornado.api.TornadoExecutionPlan
import uk.ac.manchester.tornado.api.enums.DataTransferMode
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException
import java.util.*
import kotlin.math.max

typealias TvmFloatArray = uk.ac.manchester.tornado.api.types.arrays.FloatArray
typealias TvmByteArray = uk.ac.manchester.tornado.api.types.arrays.ByteArray
typealias TvmArray = uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray

class InferenceExecutionContext {
    private lateinit var singlePassMemory: ContextMemory<TvmFloatArray>
    private lateinit var operationLocalMemory: ContextMemory<TvmFloatArray>
    private lateinit var residentMemory: ContextMemory<TvmByteArray>
    private lateinit var inputMemory: ContextMemory<TvmFloatArray>

    private lateinit var terminalOperation: Operation
    private val stages = ArrayList<List<Operation>>()

    private var taskGraph: ImmutableTaskGraph? = null
    private var executionPlan: TornadoExecutionPlan? = null
    private var executionResult: TvmFloatArray? = null

    fun initializeExecution(terminalOperation: Operation) {
        checkInitialized()

        this.terminalOperation = terminalOperation

        //Find last operations for all layers and optimize the execution graph.
        splitExecutionGraphByStages()

        //For each layer calculate the maximum buffer size needed for the backward and forward  calculation.
        //For forward calculation, the buffer size is the sum of the memory requirements of all operations in the layer.
        //For backward calculation, the buffer size is maximum of the memory requirements of all operations in the layer.
        initializeBuffers()
    }

    private fun checkInitialized() {
        if (taskGraph != null) {
            throw IllegalStateException("Execution context is already initialized")
        }
    }

    fun executePass(): FloatArray {
        terminalOperation.prepareForNextExecutionPass()

        if (taskGraph == null) {
            val taskGraph = TaskGraph(TvmCommons.generateName("executionPass"))

            taskGraph.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                operationLocalMemory.memoryBuffer,
                singlePassMemory.memoryBuffer,
                residentMemory.memoryBuffer
            )

            taskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inputMemory.memoryBuffer)
            val resultPointer = terminalOperation.buildTaskGraph(taskGraph)

            val executionResult = TvmFloatArray(CommonTensorOperations.stride(resultPointer.shape))
            TvmVectorOperations.addCopyVectorTask(
                taskGraph,
                TvmCommons.generateName("copyResult"),
                getMemoryBuffer(resultPointer.pointer),
                TrainingExecutionContext.addressOffset(resultPointer.pointer),
                executionResult,
                0,
                executionResult.size
            )
            taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, executionResult)

            this.taskGraph = taskGraph.snapshot()
            this.executionPlan = TornadoExecutionPlan(this.taskGraph)
            this.executionResult = executionResult
        } else {
            require(executionPlan != null) { "Execution plan is not initialized" }
        }

        try {
            executionPlan!!.execute()
        } catch (e: TornadoExecutionPlanException) {
            throw RuntimeException("Failed to execute the task graph", e)
        }

        return executionResult!!.toHeapArray()
    }

    private fun splitExecutionGraphByStages() {
        val operations = ArrayList<Operation>()
        operations.add(terminalOperation)

        splitExecutionGraphByStages(operations)

        stages.reverse()
    }

    /**
     * Split the execution graph into layers starting from the passed operations.
     */
    private fun splitExecutionGraphByStages(operations: List<Operation>) {
        stages.add(operations)

        val nextStageOperations = ArrayList<Operation>()
        val visitedOperations = HashSet<Operation>()

        for (operation in operations) {
            val previousLeftOperation = operation.leftPreviousOperation
            val previousRightOperation = operation.rightPreviousOperation

            if (previousLeftOperation != null) {
                if (visitedOperations.add(previousLeftOperation)) {
                    nextStageOperations.add(previousLeftOperation)
                }
            }

            if (previousRightOperation != null) {
                if (visitedOperations.add(previousRightOperation)) {
                    nextStageOperations.add(previousRightOperation)
                }
            }
        }

        if (nextStageOperations.isEmpty()) {
            return
        }

        splitExecutionGraphByStages(nextStageOperations)
    }

    private fun initializeBuffers() {
        var singlePassBufferLength = 0
        var operationLocalBufferLength = 0
        var residentMemoryAllocations = 0
        var inputMemoryAllocations = 0

        for (operations in stages) {
            var localBufferLength = 0

            for (operation in operations) {
                var allocations: List<IntImmutableList?> = operation.singlePassAllocations
                singlePassBufferLength += ContextMemory.allocationsSize(allocations)
                residentMemoryAllocations += ContextMemory.allocationsSize(operation.residentAllocations)
                inputMemoryAllocations += ContextMemory.allocationsSize(operation.inputAllocations)

                allocations = operation.localAllocations
                localBufferLength += ContextMemory.allocationsSize(allocations)
            }

            operationLocalBufferLength = max(localBufferLength, operationLocalBufferLength)
        }

        operationLocalMemory =
            ContextMemory(
                TvmFloatArray(operationLocalBufferLength), OPERATION_LOCAL_MEMORY_KIND, TensorPointer.DType.F32,
                true
            )
        singlePassMemory = ContextMemory(
            TvmFloatArray(singlePassBufferLength), SINGLE_PASS_MEMORY_KIND, TensorPointer.DType.F32,
            true
        )
        residentMemory = ContextMemory(
            TvmByteArray(residentMemoryAllocations), RESIDENT_MEMORY_KIND, TensorPointer.DType.INT8,
            true
        )
        inputMemory = ContextMemory(
            TvmFloatArray(inputMemoryAllocations), INPUT_MEMORY_KIND, TensorPointer.DType.F32,
            true
        )
    }

    @Suppress("unused")
    fun allocateLocalMemory(operation: Operation, dimensions: IntImmutableList): TensorPointer {
        checkInitialized()

        return operationLocalMemory.allocate(operation, dimensions) {
            operation.localAllocations
        }
    }

    fun allocateInputMemory(operation: Operation, dimensions: IntImmutableList): TensorPointer {
        checkInitialized()

        return inputMemory.allocate(operation, dimensions) {
            operation.inputAllocations
        }
    }

    fun allocateSinglePassMemory(operation: Operation, dimensions: IntImmutableList): TensorPointer {
        checkInitialized()

        return singlePassMemory.allocate(operation, dimensions) {
            operation.singlePassAllocations
        }
    }

    fun allocateResidentMemory(operation: Operation, dimensions: IntImmutableList): TensorPointer {
        checkInitialized()

        return residentMemory.allocate(operation, dimensions) {
            operation.residentAllocations
        }
    }

    fun getMemoryBuffer(address: Long): TvmArray {
        val memoryType = memoryType(address)

        return when (memoryType) {
            MemoryType.SINGLE_PASS -> singlePassMemory.memoryBuffer
            MemoryType.OPERATION_LOCAL -> operationLocalMemory.memoryBuffer
            MemoryType.RESIDENT -> residentMemory.memoryBuffer
            MemoryType.INPUT -> inputMemory.memoryBuffer
        }
    }

    private fun memoryType(address: Long): MemoryType {
        require(!ContextMemory.isNull(address)) { "Provided address is null" }

        val memoryKind = (address ushr 61).toByte()
        if (memoryKind == SINGLE_PASS_MEMORY_KIND) {
            return MemoryType.SINGLE_PASS
        }

        if (memoryKind == OPERATION_LOCAL_MEMORY_KIND) {
            return MemoryType.OPERATION_LOCAL
        }

        if (memoryKind == RESIDENT_MEMORY_KIND) {
            return MemoryType.RESIDENT
        }

        if (memoryKind == INPUT_MEMORY_KIND) {
            return MemoryType.INPUT
        }

        throw IllegalArgumentException("Unknown memory kind : $memoryKind")
    }

    private enum class MemoryType {
        SINGLE_PASS,
        OPERATION_LOCAL,
        RESIDENT,
        INPUT
    }

    companion object {
        private const val SINGLE_PASS_MEMORY_KIND: Byte = 1
        private const val OPERATION_LOCAL_MEMORY_KIND: Byte = 2
        private const val RESIDENT_MEMORY_KIND: Byte = 3
        private const val INPUT_MEMORY_KIND: Byte = 4


        fun addressOffset(address: Long): Int {
            return ContextMemory.addressOffset(address)
        }
    }
}
