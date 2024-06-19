package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.Arrays;

public abstract class AbstractOperation implements Operation {
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
            leftOperation.setNextOperation(this);
        }

        if (rightOperation != null) {
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
                                              TensorPointer secondTensor, BiKernelFunction function) {
        var firstTensorShape = firstTensor.shape();
        var secondTensorShape = secondTensor.shape();

        var broadcastCandidate = TensorOperations.broadcastCandidate(firstTensorShape, secondTensorShape);
        if (broadcastCandidate == -1) {
            throw new IllegalArgumentException("Invalid shapes for operation. First shape: " +
                    Arrays.toString(firstTensorShape) + ", second shape: " + Arrays.toString(secondTensorShape) + ".");
        }

        if (broadcastCandidate == 0) {
            var result = executionContext.allocateForwardMemory(firstTensorShape);
            function.apply(firstTensor, secondTensor, result);
            return result;
        }

        if (broadcastCandidate == 1) {
            var temp = firstTensor;

            firstTensor = secondTensor;
            secondTensor = temp;
        }

        var broadcastTensor = executionContext.allocateForwardMemory(secondTensorShape);
        var broadcastTensorBuffer = executionContext.getMemoryBuffer(broadcastTensor.pointer());
        var broadcastTensorOffset = TrainingExecutionContext.addressOffset(broadcastTensor.pointer());

        var firstTensorBuffer = executionContext.getMemoryBuffer(firstTensor.pointer());
        var firstTensorOffset = TrainingExecutionContext.addressOffset(firstTensor.pointer());


        TensorOperations.broadcast(firstTensorBuffer, firstTensorOffset, firstTensorShape,
                broadcastTensorBuffer, broadcastTensorOffset, secondTensorShape);

        if (broadcastCandidate == 1) {
            function.apply(secondTensor, broadcastTensor, broadcastTensor);
            return broadcastTensor;
        }

        function.apply(broadcastTensor, secondTensor, broadcastTensor);
        return broadcastTensor;
    }

    protected TensorPointer reduceIfNeeded(TensorPointer firstTensor, TensorPointer secondTensor, BiKernelFunction function) {
        var firstTensorShape = firstTensor.shape();
        var secondTensorShape = secondTensor.shape();

        var broadcastCandidate = TensorOperations.broadcastCandidate(firstTensorShape, secondTensorShape);
        if (broadcastCandidate == -1) {
            throw new IllegalArgumentException("Invalid shapes for operation. First shape: " +
                    Arrays.toString(firstTensorShape) + ", second shape: " + Arrays.toString(secondTensorShape) + ".");
        }

        if (broadcastCandidate == 0) {
            var result = executionContext.allocateForwardMemory(firstTensorShape);
            function.apply(firstTensor, secondTensor, result);
            return result;
        }

        if (broadcastCandidate == 1) {
            var temp = secondTensor;

            secondTensor = firstTensor;
            firstTensor = temp;
        }

        var reducedTensor = executionContext.allocateForwardMemory(firstTensorShape);
        var reducedTensorBuffer = executionContext.getMemoryBuffer(reducedTensor.pointer());
        var reducedTensorOffset = TrainingExecutionContext.addressOffset(reducedTensor.pointer());

        var firstTensorBuffer = executionContext.getMemoryBuffer(firstTensor.pointer());
        var firstTensorOffset = TrainingExecutionContext.addressOffset(firstTensor.pointer());

        TensorOperations.reduce(firstTensorBuffer, firstTensorOffset, firstTensorShape,
                reducedTensorBuffer, reducedTensorOffset, secondTensorShape);

        if (broadcastCandidate == 1) {
            function.apply(secondTensor, reducedTensor, reducedTensor);
            return reducedTensor;
        }

        function.apply(reducedTensor, secondTensor, reducedTensor);
        return reducedTensor;
    }

    protected interface BiKernelFunction {
        void apply(TensorPointer firstTensor, TensorPointer secondTensor, TensorPointer result);
    }
}
