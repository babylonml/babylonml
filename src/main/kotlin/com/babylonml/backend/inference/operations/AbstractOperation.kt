package com.babylonml.backend.inference.operations

import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.inference.operations.tornadovm.InferenceExecutionContext
import com.babylonml.backend.inference.operations.tornadovm.TvmFloatArray
import com.babylonml.backend.tornadovm.TvmCommons
import com.babylonml.backend.tornadovm.TvmTensorOperations
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.TaskGraph

abstract class AbstractOperation(
    override val name: String?, executionContext: InferenceExecutionContext?,
    final override var leftPreviousOperation: Operation?, final override var rightPreviousOperation: Operation?
) : Operation {
    protected val singlePassMemoryAllocator: (operation: Operation, shape: IntImmutableList) -> TensorPointer
    protected val localMemoryAllocator: (operation: Operation, shape: IntImmutableList) -> TensorPointer

    final override var executionContext: InferenceExecutionContext

    override var nextOperation: Operation? = null

    constructor(
        name: String?,
        leftOperation: Operation?,
        rightOperation: Operation?
    ) : this(name, null, leftOperation, rightOperation)

    constructor(leftOperation: Operation?, rightOperation: Operation?) : this(
        null,
        null,
        leftOperation,
        rightOperation
    )

    constructor(
        executionContext: InferenceExecutionContext?,
        leftOperation: Operation?, rightOperation: Operation?
    ) : this(null, executionContext, leftOperation, rightOperation)

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

        val ex = this.executionContext
        singlePassMemoryAllocator = { operation, shape ->
            ex.allocateSinglePassMemory(
                operation,
                shape
            )
        }
        localMemoryAllocator = { operation, shape ->
            ex.allocateLocalMemory(
                operation,
                shape
            )
        }
    }

    override fun prepareForNextPass() {
        leftPreviousOperation?.prepareForNextPass()
        rightPreviousOperation?.prepareForNextPass()
    }


    protected fun broadcastIfNeeded(
        taskGraph: TaskGraph,
        firstTensor: TensorPointer,
        secondTensor: TensorPointer,
        function: (firstTensor: TensorPointer, secondTensor: TensorPointer, result: TensorPointer) -> Unit
    ): TensorPointer {
        val firstTensorShape = firstTensor.shape
        val secondTensorShape = secondTensor.shape

        val broadcastCandidate = TensorOperations.broadcastCandidate(firstTensorShape, secondTensorShape)
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
                taskGraph, getTaskName("BroadcastIfNeeded"),
                secondTensor.buffer(), secondTensor.offset(), secondTensor.shape,
                broadcastTensor.buffer(), broadcastTensor.offset(), broadcastTensor.shape
            )
            function(broadcastTensor, secondTensor, broadcastTensor)
            return broadcastTensor
        }

        val broadcastTensor = executionContext.allocateSinglePassMemory(this, firstTensorShape)
        TvmTensorOperations.addBroadcastTask(
            taskGraph, getTaskName("BroadcastIfNeeded"),
            firstTensor.buffer(), firstTensor.offset(), firstTensor.shape,
            broadcastTensor.buffer(), broadcastTensor.offset(), broadcastTensor.shape
        )
        function(firstTensor, broadcastTensor, broadcastTensor)

        return broadcastTensor
    }

    fun TensorPointer.buffer(): TvmFloatArray {
        return executionContext.getMemoryBuffer(pointer)
    }

    fun TensorPointer.offset() = InferenceExecutionContext.addressOffset(pointer)

    fun getTaskName(prefix: String? = null): String {
        return TvmCommons.generateName(
            if (name == null) {
                prefix ?: ("" + this::class.simpleName)
            } else {
                prefix ?: ("" + name)
            }
        )
    }
}
