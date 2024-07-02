package com.babylonml.backend.inference.operations.tornadovm

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.inference.operations.Operation
import com.babylonml.backend.inference.tornadovm.ContextMemory
import com.babylonml.backend.inference.tornadovm.InputSource
import com.babylonml.backend.training.execution.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList
import org.jspecify.annotations.Nullable
import uk.ac.manchester.tornado.api.TaskGraph
import uk.ac.manchester.tornado.api.TornadoExecutionPlan
import uk.ac.manchester.tornado.api.annotations.Parallel
import uk.ac.manchester.tornado.api.enums.DataTransferMode
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException
import uk.ac.manchester.tornado.api.types.tensors.Tensor
import java.util.*
import kotlin.math.max

typealias TvmFloatArray = uk.ac.manchester.tornado.api.types.arrays.FloatArray

class InferenceExecutionContext {
    private lateinit var singlePassMemory: ContextMemory
    private lateinit var operationLocalMemory: ContextMemory
    private lateinit var terminalOperation: Operation
    private val stages = ArrayList<List<Operation>>()

    private var inputSource: @Nullable InputSource? = null

    fun registerMainInputSource(data: Tensor): InputSource {
        check(inputSource == null) { "Input source is already registered" }

        val inputSource = TensorInputSource(data)
        this.inputSource = inputSource

        return inputSource
    }

    fun registerAdditionalInputSource(data: Tensor?): InputSource {
        checkNotNull(inputSource) { "Main input source is not registered" }
        return TensorInputSource(data!!)
    }

    fun initializeExecution(terminalOperation: Operation) {
        this.terminalOperation = terminalOperation

        //Find last operations for all layers and optimize the execution graph.
        splitExecutionGraphByStages()

        //For each layer calculate the maximum buffer size needed for the backward and forward  calculation.
        //For forward calculation, the buffer size is the sum of the memory requirements of all operations in the layer.
        //For backward calculation, the buffer size is maximum of the memory requirements of all operations in the layer.
        initializeBuffers()
    }

    fun executePass(): FloatArray {
        terminalOperation.prepareForNextPass()

        val taskGraph = TaskGraph("executionPass")
        taskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, operationLocalMemory, singlePassMemory)
        val resultPointer = terminalOperation.execute(taskGraph)

        val result = TvmFloatArray(CommonTensorOperations.stride(resultPointer.shape))
        taskGraph.task(
            "fetchExecutionResult",
            { singlePassMemoryBuffer: TvmFloatArray, res: TvmFloatArray ->
                fetchExecutionResult(
                    singlePassMemoryBuffer,
                    TrainingExecutionContext.addressOffset(resultPointer.pointer),
                    res
                )
            },
            singlePassMemory.memoryBuffer,
            result
        )
        taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, result)

        val immutableTaskGraph = taskGraph.snapshot()
        try {
            TornadoExecutionPlan(immutableTaskGraph).use { executionPlan ->
                executionPlan.execute()
            }
        } catch (e: TornadoExecutionPlanException) {
            throw RuntimeException("Failed to execute the task graph", e)
        }

        return result.toHeapArray()
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

        for (operations in stages) {
            var localBufferLength = 0

            for (operation in operations) {
                var allocations: List<IntImmutableList?> = operation.singlePassAllocations
                singlePassBufferLength += ContextMemory.allocationsSize(allocations)

                allocations = operation.localAllocations
                localBufferLength += ContextMemory.allocationsSize(allocations)
            }

            operationLocalBufferLength = max(localBufferLength.toDouble(), operationLocalBufferLength.toDouble())
                .toInt()
        }

        operationLocalMemory = ContextMemory(operationLocalBufferLength, OPERATION_LOCAL_MEMORY_TYPE, true)
        singlePassMemory = ContextMemory(singlePassBufferLength, SINGLE_PASS_MEMORY_TYPE, true)
    }

    fun allocateLocalMemory(operation: Operation, dimensions: IntImmutableList): TensorPointer {
        return operationLocalMemory.allocate(operation, dimensions) {
            operation.localAllocations
        }
    }

    fun allocateSinglePassMemory(operation: Operation, dimensions: IntImmutableList): TensorPointer {
        return singlePassMemory.allocate(operation, dimensions) {
            operation.singlePassAllocations
        }
    }

    fun getMemoryBuffer(address: Long): TvmFloatArray {
        val memoryType = memoryType(address)

        return when (memoryType) {
            MemoryType.SINGLE_PASS -> singlePassMemory.memoryBuffer
            MemoryType.OPERATION_LOCAL -> operationLocalMemory.memoryBuffer
        }
    }

    private fun memoryType(address: Long): MemoryType {
        require(!ContextMemory.isNull(address)) { "Provided address is null" }

        val memoryType = address ushr 62
        if (memoryType == SINGLE_PASS_MEMORY_TYPE.toLong()) {
            return MemoryType.SINGLE_PASS
        }

        if (memoryType == OPERATION_LOCAL_MEMORY_TYPE.toLong()) {
            return MemoryType.OPERATION_LOCAL
        }

        throw IllegalArgumentException("Unknown memory type : $memoryType")
    }

    private enum class MemoryType {
        SINGLE_PASS,
        OPERATION_LOCAL
    }

    companion object {
        private const val SINGLE_PASS_MEMORY_TYPE: Byte = 1
        private const val OPERATION_LOCAL_MEMORY_TYPE: Byte = 2

        @JvmStatic
        fun fetchExecutionResult(
            singlePassMemoryBuffer: TvmFloatArray,
            offset: Int,
            result: TvmFloatArray
        ) {
            for (i: @Parallel Int in 0 until result.size) {
                result[i] = singlePassMemoryBuffer[offset + i]
            }
        }

        fun addressOffset(address: Long): Int {
            return ContextMemory.addressOffset(address)
        }
    }
}
