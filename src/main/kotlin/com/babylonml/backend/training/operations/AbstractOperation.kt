package com.babylonml.backend.training.operations

import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList
import java.util.function.BiFunction

abstract class AbstractOperation(
    protected var name: String?, executionContext: TrainingExecutionContext?,
    final override var leftPreviousOperation: Operation?, final override var rightPreviousOperation: Operation?
) : Operation {
    protected val forwardMemoryAllocator: BiFunction<Operation, IntImmutableList, TensorPointer>
    protected val backwardMemoryAllocator: BiFunction<Operation, IntImmutableList, TensorPointer>

    final override var executionContext: TrainingExecutionContext

    override var derivativeChainPointer: TensorPointer? = null

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
        executionContext: TrainingExecutionContext?,
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
        forwardMemoryAllocator = BiFunction { operation: Operation, dimensions: IntImmutableList ->
            ex.allocateForwardMemory(
                operation,
                dimensions
            )
        }
        backwardMemoryAllocator = BiFunction { operation: Operation, dimensions: IntImmutableList ->
            ex.allocateBackwardMemory(
                operation,
                dimensions
            )
        }
    }

    override fun clearNextOperation() {
        this.nextOperation = null
    }

    override fun prepareForNextPropagation() {
        leftPreviousOperation?.prepareForNextPropagation()
        rightPreviousOperation?.prepareForNextPropagation()
    }

    override fun startEpochExecution() {
        leftPreviousOperation?.startEpochExecution()
        rightPreviousOperation?.startEpochExecution()
    }


    protected fun broadcastIfNeeded(
        firstTensor: TensorPointer,
        secondTensor: TensorPointer,
        allocator: BiFunction<Operation, IntImmutableList, TensorPointer>,
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
            val result = allocator.apply(this, firstTensorShape)
            function(firstTensor, secondTensor, result)
            return result
        }

        if (broadcastCandidate == 1) {
            val broadcastTensor = allocator.apply(this, secondTensorShape)
            TensorOperations.broadcast(
                firstTensor.buffer(), firstTensor.offset(), firstTensor.shape,
                broadcastTensor.buffer(), broadcastTensor.offset(), broadcastTensor.shape
            )
            function(broadcastTensor, secondTensor, broadcastTensor)
            return broadcastTensor
        }

        val broadcastTensor = allocator.apply(this, firstTensorShape)
        TensorOperations.broadcast(
            secondTensor.buffer(), secondTensor.offset(), secondTensor.shape,
            broadcastTensor.buffer(), broadcastTensor.offset(), broadcastTensor.shape
        )

        function(firstTensor, broadcastTensor, broadcastTensor)

        return broadcastTensor
    }

    protected fun reduceIfNeeded(
        firstTensor: TensorPointer, secondTensor: TensorPointer,
        allocator: BiFunction<Operation, IntImmutableList, TensorPointer>,
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
            val result = allocator.apply(this, firstTensorShape)
            function(firstTensor, secondTensor, result)

            return result
        }

        if (broadcastCandidate == 1) {
            val reducedTensor = allocator.apply(this, firstTensorShape)
            TensorOperations.reduce(
                secondTensor.buffer(), secondTensor.offset(), secondTensor.shape,
                reducedTensor.buffer(), reducedTensor.offset(), reducedTensor.shape
            )
            function(firstTensor, reducedTensor, reducedTensor)

            return reducedTensor
        }

        val reducedTensor = allocator.apply(this, secondTensorShape)
        TensorOperations.reduce(
            firstTensor.buffer(), firstTensor.offset(), firstTensor.shape,
            reducedTensor.buffer(), reducedTensor.offset(), reducedTensor.shape
        )
        function(reducedTensor, secondTensor, reducedTensor)

        return reducedTensor
    }
}
