package com.babylonml.backend.inference.operations

import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.inference.operations.tornadovm.InferenceExecutionContext
import com.babylonml.backend.inference.operations.tornadovm.TvmArray
import com.babylonml.backend.inference.operations.tornadovm.TvmFloatArray
import com.babylonml.backend.tornadovm.TvmCommons
import com.babylonml.backend.tornadovm.TvmTensorOperations
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.TaskGraph

abstract class AbstractOperation(
    override val name: String,
    executionContext: InferenceExecutionContext?,
    final override var leftPreviousOperation: Operation?,
    final override var rightPreviousOperation: Operation?
) : Operation {
    final override var executionContext: InferenceExecutionContext

    override val singlePassAllocations: List<IntImmutableList>
        get() = emptyList()
    override val localAllocations: List<IntImmutableList>
        get() = emptyList()
    override val inputAllocations: List<IntImmutableList>
        get() = emptyList()
    override val residentInt8Allocations: List<IntImmutableList>
        get() = emptyList()
    override val residentF16Allocations: List<IntImmutableList>
        get() = emptyList()
    override val residentF32Allocations: List<IntImmutableList>
        get() = emptyList()

    override var nextOperation: Operation? = null

    constructor(
        name: String,
        leftOperation: Operation?,
        rightOperation: Operation?
    ) : this(name, null, leftOperation, rightOperation)

    init {
        leftPreviousOperation?.let {
            require(it.nextOperation == null) { "Left operation already has a next operation" }
            it.nextOperation = this
        }

        rightPreviousOperation?.let {
            require(it.nextOperation == null) { "Right operation already has a next operation" }
            it.nextOperation = this
        }

        if (executionContext == null) {
            if (leftPreviousOperation != null) {
                this.executionContext = leftPreviousOperation!!.executionContext
            } else if (rightPreviousOperation != null) {
                this.executionContext = rightPreviousOperation!!.executionContext
            } else {
                throw IllegalArgumentException("At least one of the operations should be provided")
            }
        } else {
            this.executionContext = executionContext
        }
    }

    final override fun buildTaskGraph(taskGraph: TaskGraph): TensorPointer {
        prepareForNextExecutionPass()
        return doBuildTaskGraph(taskGraph)
    }

    abstract fun doBuildTaskGraph(taskGraph: TaskGraph): TensorPointer

    override fun prepareForNextExecutionPass() {
        leftPreviousOperation?.prepareForNextExecutionPass()
        rightPreviousOperation?.prepareForNextExecutionPass()
    }

    protected fun broadcastIfNeeded(
        taskGraph: TaskGraph,
        firstTensor: TensorPointer,
        secondTensor: TensorPointer,
        function: (firstTensor: TensorPointer, secondTensor: TensorPointer, result: TensorPointer) -> Unit
    ): TensorPointer {
        val firstTensorShape = firstTensor.shape
        val secondTensorShape = secondTensor.shape

        val broadcastCandidate =
            TensorOperations.broadcastCandidate(firstTensorShape, secondTensorShape)
        require(broadcastCandidate != -1) {
            "Invalid shapes for operation. First shape: " +
                    firstTensorShape + ", second shape: " + secondTensorShape + "."
        }

        if (broadcastCandidate == 0) {
            val result = executionContext.allocateSinglePassMemory(this, firstTensorShape)
            function(firstTensor, secondTensor, result)
            return result
        }

        if (broadcastCandidate == 1) {
            val broadcastTensor = executionContext.allocateSinglePassMemory(this, secondTensorShape)
            TvmTensorOperations.addBroadcastTask(
                taskGraph,
                getTaskName("BroadcastIfNeeded"),
                secondTensor.buffer() as TvmFloatArray,
                secondTensor.offset(),
                secondTensor.shape,
                broadcastTensor.buffer() as TvmFloatArray,
                broadcastTensor.offset(),
                broadcastTensor.shape
            )
            function(broadcastTensor, secondTensor, broadcastTensor)
            return broadcastTensor
        }

        val broadcastTensor = executionContext.allocateSinglePassMemory(this, firstTensorShape)
        TvmTensorOperations.addBroadcastTask(
            taskGraph,
            getTaskName("BroadcastIfNeeded"),
            firstTensor.buffer() as TvmFloatArray,
            firstTensor.offset(),
            firstTensor.shape,
            broadcastTensor.buffer() as TvmFloatArray,
            broadcastTensor.offset(),
            broadcastTensor.shape
        )
        function(firstTensor, broadcastTensor, broadcastTensor)

        return broadcastTensor
    }

    fun TensorPointer.buffer(): TvmArray {
        return executionContext.getMemoryBuffer(this)
    }

    fun TensorPointer.floatBuffer(): TvmFloatArray {
        if (dtype != TensorPointer.DType.F32) {
            throw IllegalArgumentException("Tensor is not of type F32")
        }

        return buffer() as TvmFloatArray
    }

    fun TensorPointer.offset() = pointer.toInt()

    fun getTaskName(prefix: String? = null): String {
        return TvmCommons.generateName(
            prefix ?: ("" + name)
        )
    }
}
