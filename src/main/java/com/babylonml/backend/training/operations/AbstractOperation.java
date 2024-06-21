package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.BiFunction;

public abstract class AbstractOperation implements Operation {
    protected final BiFunction<Operation, int[], TensorPointer> forwardMemoryAllocator;
    protected final BiFunction<Operation, int[], TensorPointer> backwardMemoryAllocator;

    protected Operation leftOperation;
    protected Operation rightOperation;

    @NonNull
    protected final TrainingExecutionContext executionContext;

    @Nullable
    protected TensorPointer derivativeChainPointer;

    protected Operation nextOperation;

    @Nullable
    protected String name;

    public AbstractOperation(String name, Operation leftOperation, Operation rightOperation) {
        this(name, null, leftOperation, rightOperation);
    }

    public AbstractOperation(Operation leftOperation, Operation rightOperation) {
        this(null, null, leftOperation, rightOperation);
    }

    public AbstractOperation(TrainingExecutionContext executionContext,
                             Operation leftOperation, Operation rightOperation) {
        this(null, executionContext, leftOperation, rightOperation);
    }


    public AbstractOperation(@Nullable String name, @Nullable TrainingExecutionContext executionContext,
                             Operation leftOperation, Operation rightOperation) {
        this.name = name;

        this.leftOperation = leftOperation;
        this.rightOperation = rightOperation;

        if (leftOperation != null) {
            if (leftOperation.getNextOperation() != null) {
                throw new IllegalArgumentException("Left operation already has a next operation");
            }

            leftOperation.setNextOperation(this);
        }

        if (rightOperation != null) {
            if (rightOperation.getNextOperation() != null) {
                throw new IllegalArgumentException("Right operation already has a next operation");
            }

            rightOperation.setNextOperation(this);
        }

        if (executionContext == null) {
            if (leftOperation != null) {
                this.executionContext = leftOperation.getExecutionContext();
            } else if (rightOperation != null) {
                this.executionContext = rightOperation.getExecutionContext();
            } else {
                throw new IllegalArgumentException("At least one of the operations should be provided");
            }
        } else {
            this.executionContext = executionContext;
        }

        Objects.requireNonNull(this.executionContext);

        var ex = this.executionContext;

        forwardMemoryAllocator = ex::allocateForwardMemory;
        backwardMemoryAllocator = ex::allocateBackwardMemory;
    }

    public final void updateBackwardDerivativeChainValue(@NonNull TensorPointer backwardDerivativeChainValue) {
        this.derivativeChainPointer = backwardDerivativeChainValue;
    }

    @Override
    public final Operation getLeftPreviousOperation() {
        return leftOperation;
    }

    @Override
    public final Operation getRightPreviousOperation() {
        return rightOperation;
    }

    @Override
    public final void setLeftPreviousOperation(@NonNull Operation leftPreviousOperation) {
        this.leftOperation = leftPreviousOperation;
    }

    @Override
    public final void setRightPreviousOperation(@NonNull Operation rightPreviousOperation) {
        this.rightOperation = rightPreviousOperation;
    }

    @Override
    public Operation getNextOperation() {
        return nextOperation;
    }

    @Override
    public final void setNextOperation(@NonNull Operation nextOperation) {
        this.nextOperation = nextOperation;
    }

    @Override
    public void clearNextOperation() {
        this.nextOperation = null;
    }

    @Override
    public void prepareForNextPropagation() {
        if (leftOperation != null) {
            leftOperation.prepareForNextPropagation();
        }

        if (rightOperation != null) {
            rightOperation.prepareForNextPropagation();
        }
    }

    @Override
    public void startEpochExecution() {
        if (leftOperation != null) {
            leftOperation.startEpochExecution();
        }

        if (rightOperation != null) {
            rightOperation.startEpochExecution();
        }
    }

    @Override
    public @NonNull TrainingExecutionContext getExecutionContext() {
        return executionContext;
    }


    protected TensorPointer broadcastIfNeeded(TensorPointer firstTensor,
                                              TensorPointer secondTensor,
                                              BiFunction<Operation, int[], TensorPointer> allocator,
                                              BiKernelFunction function) {
        var firstTensorShape = firstTensor.shape();
        var secondTensorShape = secondTensor.shape();

        var broadcastCandidate = TensorOperations.broadcastCandidate(firstTensorShape, secondTensorShape);
        if (broadcastCandidate == -1) {
            throw new IllegalArgumentException("Invalid shapes for operation. First shape: " +
                    Arrays.toString(firstTensorShape) + ", second shape: " + Arrays.toString(secondTensorShape) + ".");
        }

        if (broadcastCandidate == 0) {
            var result = allocator.apply(this, firstTensorShape);
            function.apply(firstTensor, secondTensor, result);
            return result;
        }

        if (broadcastCandidate == 1) {
            var broadcastTensor = allocator.apply(this, secondTensorShape);
            TensorOperations.broadcast(firstTensor.buffer(), firstTensor.offset(), firstTensor.shape(),
                    broadcastTensor.buffer(), broadcastTensor.offset(), broadcastTensor.shape());
            function.apply(broadcastTensor, secondTensor, broadcastTensor);
            return broadcastTensor;
        }

        var broadcastTensor = allocator.apply(this, firstTensorShape);
        TensorOperations.broadcast(secondTensor.buffer(), secondTensor.offset(), secondTensor.shape(),
                broadcastTensor.buffer(), broadcastTensor.offset(), broadcastTensor.shape());

        function.apply(firstTensor, broadcastTensor, broadcastTensor);

        return broadcastTensor;
    }

    protected TensorPointer reduceIfNeeded(TensorPointer firstTensor, TensorPointer secondTensor,
                                           BiFunction<Operation, int[], TensorPointer> allocator,
                                           BiKernelFunction function) {
        var firstTensorShape = firstTensor.shape();
        var secondTensorShape = secondTensor.shape();

        var broadcastCandidate = TensorOperations.broadcastCandidate(firstTensorShape, secondTensorShape);
        if (broadcastCandidate == -1) {
            throw new IllegalArgumentException("Invalid shapes for operation. First shape: " +
                    Arrays.toString(firstTensorShape) + ", second shape: " + Arrays.toString(secondTensorShape) + ".");
        }

        if (broadcastCandidate == 0) {
            var result = allocator.apply(this, firstTensorShape);
            function.apply(firstTensor, secondTensor, result);

            return result;
        }

        if (broadcastCandidate == 1) {
            var reducedTensor = allocator.apply(this, firstTensorShape);
            TensorOperations.reduce(secondTensor.buffer(), secondTensor.offset(), secondTensor.shape(),
                    reducedTensor.buffer(), reducedTensor.offset(), reducedTensor.shape());
            function.apply(firstTensor, reducedTensor, reducedTensor);

            return reducedTensor;
        }

        var reducedTensor = allocator.apply(this, secondTensorShape);
        TensorOperations.reduce(firstTensor.buffer(), firstTensor.offset(), firstTensor.shape(),
                reducedTensor.buffer(), reducedTensor.offset(), reducedTensor.shape());
        function.apply(reducedTensor, secondTensor, reducedTensor);

        return reducedTensor;
    }

    protected interface BiKernelFunction {
        void apply(TensorPointer firstTensor, TensorPointer secondTensor, TensorPointer result);
    }
}
