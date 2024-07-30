package com.babylonml.backend

import com.babylonml.backend.operations.*
import com.babylonml.backend.tensor.common.CommonTensorOperations
import com.babylonml.backend.tensor.common.TensorPointer
import com.babylonml.backend.tensor.tornadovm.TvmCommons
import com.babylonml.backend.tensor.tornadovm.TvmVectorOperations
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.GridScheduler
import uk.ac.manchester.tornado.api.TaskGraph
import uk.ac.manchester.tornado.api.TornadoExecutionPlan
import uk.ac.manchester.tornado.api.enums.DataTransferMode
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException
import java.util.ArrayList
import java.util.HashSet
import kotlin.math.max

typealias TvmFloatArray = uk.ac.manchester.tornado.api.types.arrays.FloatArray
typealias TvmByteArray = uk.ac.manchester.tornado.api.types.arrays.ByteArray
typealias TvmIntArray = uk.ac.manchester.tornado.api.types.arrays.IntArray
typealias TvmHalfFloatArray = uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray
typealias TvmArray = uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray


class ExecutionContext : AutoCloseable {
    private lateinit var singlePassMemory: ContextMemory<TvmFloatArray>
    private lateinit var operationLocalMemory: ContextMemory<TvmFloatArray>

    private lateinit var residentMemoryInt8: ContextMemory<TvmByteArray>
    private lateinit var residentMemoryF16: ContextMemory<TvmHalfFloatArray>
    private lateinit var residentMemoryF32: ContextMemory<TvmFloatArray>

    private lateinit var inputMemoryF32: ContextMemory<TvmFloatArray>
    private lateinit var inputMemoryI32: ContextMemory<TvmIntArray>

    private lateinit var terminalOperation: AbstractOperation
    private val stages = ArrayList<List<AbstractOperation>>()


    private var executionPlan: TornadoExecutionPlan? = null
    private var executionResult: TvmFloatArray? = null

    fun initializeExecution(terminalOperation: AbstractOperation) {
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
        if (executionPlan != null) {
            throw IllegalStateException("Execution context is already initialized")
        }
    }

    fun executePass(): FloatArray {
        terminalOperation.prepareForNextExecutionPass()

        if (executionPlan == null) {

            val taskGraph = TaskGraph(TvmCommons.generateName("executionPass"))
            val gridScheduler = GridScheduler()

            taskGraph.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                operationLocalMemory.memoryBuffer,
                singlePassMemory.memoryBuffer,
                residentMemoryInt8.memoryBuffer,
                residentMemoryF16.memoryBuffer,
                residentMemoryF32.memoryBuffer
            )

            taskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inputMemoryF32.memoryBuffer)
            val resultPointer = terminalOperation.buildTaskGraph(taskGraph, gridScheduler)

            val executionResult = TvmFloatArray(CommonTensorOperations.stride(resultPointer.shape))
            TvmVectorOperations.addCopyVectorTask(
                taskGraph,
                TvmCommons.generateName("copyResult"),
                gridScheduler,
                getMemoryBuffer(resultPointer),
                resultPointer.pointer.toInt(),
                executionResult,
                0,
                executionResult.size
            )
            taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, executionResult)

            val immutableTaskGraph = taskGraph.snapshot()
            this.executionPlan = TornadoExecutionPlan(immutableTaskGraph).withGridScheduler(gridScheduler)
            this.executionResult = executionResult
        }

        try {
            executionPlan!!.execute()
        } catch (e: TornadoExecutionPlanException) {
            throw RuntimeException("Failed to execute the task graph", e)
        }

        return executionResult!!.toHeapArray()
    }

    private fun splitExecutionGraphByStages() {
        val operations = ArrayList<AbstractOperation>()
        operations.add(terminalOperation)

        splitExecutionGraphByStages(operations)

        stages.reverse()
    }

    /**
     * Split the execution graph into layers starting from the passed operations.
     */
    private fun splitExecutionGraphByStages(operations: List<AbstractOperation>) {
        stages.add(operations)

        val nextStageOperations = ArrayList<AbstractOperation>()
        val visitedOperations = HashSet<AbstractOperation>()

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

        var inputF32MemoryAllocations = 0
        var inputI8MemoryAllocations = 0

        var residentMemoryInt8Allocations = 0
        var residentMemoryF32Allocations = 0
        var residentMemoryF16Allocations = 0

        for (operations in stages) {
            var localBufferLength = 0

            for (operation in operations) {
                var allocations: List<IntImmutableList> = operation.maxSinglePassAllocations
                singlePassBufferLength += ContextMemory.allocationsSize(allocations)

                residentMemoryInt8Allocations += ContextMemory.allocationsSize(operation.maxResidentInt8Allocations)
                residentMemoryF32Allocations += ContextMemory.allocationsSize(operation.maxResidentF32Allocations)
                residentMemoryF16Allocations += ContextMemory.allocationsSize(operation.maxResidentF16Allocations)

                inputF32MemoryAllocations += ContextMemory.allocationsSize(operation.maxF32InputAllocations)
                inputI8MemoryAllocations += ContextMemory.allocationsSize(operation.maxI32InputAllocations)

                allocations = operation.maxLocalAllocations
                localBufferLength += ContextMemory.allocationsSize(allocations)
            }

            operationLocalBufferLength = max(localBufferLength, operationLocalBufferLength)
        }

        operationLocalMemory =
            ContextMemory(
                TvmFloatArray(operationLocalBufferLength),
                TensorPointer.MemoryKind.OPERATION_LOCAL,
                TensorPointer.DType.F32,
                true
            )

        singlePassMemory = ContextMemory(
            TvmFloatArray(singlePassBufferLength),
            TensorPointer.MemoryKind.SINGLE_PASS,
            TensorPointer.DType.F32,
            true
        )

        residentMemoryInt8 = ContextMemory(
            TvmByteArray(residentMemoryInt8Allocations),
            TensorPointer.MemoryKind.RESIDENT,
            TensorPointer.DType.INT8,
            true
        )
        residentMemoryF32 = ContextMemory(
            TvmFloatArray(residentMemoryF32Allocations),
            TensorPointer.MemoryKind.RESIDENT,
            TensorPointer.DType.F32,
            true
        )
        residentMemoryF16 = ContextMemory(
            TvmHalfFloatArray(residentMemoryF16Allocations),
            TensorPointer.MemoryKind.RESIDENT,
            TensorPointer.DType.F16,
            true
        )

        inputMemoryF32 = ContextMemory(
            TvmFloatArray(inputF32MemoryAllocations),
            TensorPointer.MemoryKind.INPUT,
            TensorPointer.DType.F32,
            true
        )
        inputMemoryI32 = ContextMemory(
            TvmIntArray(inputI8MemoryAllocations),
            TensorPointer.MemoryKind.INPUT,
            TensorPointer.DType.INT32,
            true
        )
    }

    @Suppress("unused")
    fun allocateLocalMemory(operation: AbstractOperation, dimensions: IntImmutableList): TensorPointer {
        checkInitialized()

        return operationLocalMemory.allocate(operation, dimensions) {
            operation.maxLocalAllocations
        }
    }

    fun allocateInputMemory(
        operation: AbstractOperation,
        dimensions: IntImmutableList,
        type: TensorPointer.DType
    ): TensorPointer {
        checkInitialized()

        if (type == TensorPointer.DType.F32) {
            return inputMemoryF32.allocate(operation, dimensions) {
                operation.maxF32InputAllocations
            }
        }
        if (type == TensorPointer.DType.INT32) {
            return inputMemoryI32.allocate(operation, dimensions) {
                operation.maxI32InputAllocations
            }
        }

        throw IllegalArgumentException("Unsupported tensor type: $type")
    }

    fun allocateSinglePassMemory(
        operation: AbstractOperation,
        dimensions: IntImmutableList
    ): TensorPointer {
        checkInitialized()

        return singlePassMemory.allocate(operation, dimensions) {
            operation.maxSinglePassAllocations
        }
    }

    fun allocateResidentMemory(
        operation: AbstractOperation,
        dimensions: IntImmutableList,
        type: TensorPointer.DType
    ): TensorPointer {
        checkInitialized()

        return when (type) {
            TensorPointer.DType.INT8 -> residentMemoryInt8.allocate(operation, dimensions) {
                operation.maxResidentInt8Allocations
            }

            TensorPointer.DType.F32 -> residentMemoryF32.allocate(operation, dimensions) {
                operation.maxResidentF32Allocations
            }

            TensorPointer.DType.F16 -> residentMemoryF16.allocate(operation, dimensions) {
                operation.maxResidentF16Allocations
            }

            else -> throw IllegalArgumentException("Unsupported tensor type: $type")
        }
    }

    fun getMemoryBuffer(pointer: TensorPointer): TvmArray {
        return when (pointer.memoryKind) {
            TensorPointer.MemoryKind.SINGLE_PASS -> {
                if (pointer.dtype == TensorPointer.DType.F32) {
                    singlePassMemory.memoryBuffer
                } else {
                    throw IllegalArgumentException("Unsupported tensor type: ${pointer.dtype}")
                }
            }

            TensorPointer.MemoryKind.OPERATION_LOCAL -> {
                if (pointer.dtype == TensorPointer.DType.F32) {
                    operationLocalMemory.memoryBuffer
                } else {
                    throw IllegalArgumentException("Unsupported tensor type: ${pointer.dtype}")
                }
            }

            TensorPointer.MemoryKind.INPUT -> {
                when (pointer.dtype) {
                    TensorPointer.DType.F32 -> {
                        inputMemoryF32.memoryBuffer
                    }

                    TensorPointer.DType.INT32 -> {
                        inputMemoryI32.memoryBuffer
                    }

                    else -> {
                        throw IllegalArgumentException("Unsupported tensor type: ${pointer.dtype}")
                    }
                }
            }

            TensorPointer.MemoryKind.RESIDENT -> {
                when (pointer.dtype) {
                    TensorPointer.DType.INT8 -> residentMemoryInt8.memoryBuffer
                    TensorPointer.DType.F32 -> residentMemoryF32.memoryBuffer
                    TensorPointer.DType.F16 -> residentMemoryF16.memoryBuffer
                    else -> throw IllegalArgumentException("Unsupported tensor type: ${pointer.dtype}")
                }
            }

            else -> throw IllegalArgumentException("Unsupported memory kind: ${pointer.memoryKind}")
        }
    }

    override fun close() {
        executionPlan?.close()
    }


}