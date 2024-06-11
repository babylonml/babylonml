package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

public abstract class AbstractOperation implements Operation {
    protected Operation leftOperation;
    protected Operation rightOperation;

    protected final TrainingExecutionContext executionContext;

    protected long derivativeChainPointer;

    protected Operation nextOperation;
    private int layerIndex = -1;

    @Nullable
    protected String name;

    public AbstractOperation(TrainingExecutionContext executionContext,
                             Operation leftOperation, Operation rightOperation) {
        this(null, executionContext, leftOperation, rightOperation);
    }


    public AbstractOperation(@Nullable String name, @NonNull TrainingExecutionContext executionContext,
                             Operation leftOperation, Operation rightOperation) {
        this.name = name;

        this.leftOperation = leftOperation;
        this.rightOperation = rightOperation;

        this.executionContext = executionContext;

        if (leftOperation != null) {
            leftOperation.setNextOperation(this);
        }

        if (rightOperation != null) {
            rightOperation.setNextOperation(this);
        }
    }


    public final void updateBackwardDerivativeChainValue(long backwardDerivativeChainValue) {
        this.derivativeChainPointer = backwardDerivativeChainValue;
    }

    @Override
    public final int getLayerIndex() {
        return layerIndex;
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
    public final void setLeftPreviousOperation(Operation leftPreviousOperation) {
        this.leftOperation = leftPreviousOperation;
    }

    @Override
    public final void setRightPreviousOperation(Operation rightPreviousOperation) {
        this.rightOperation = rightPreviousOperation;
    }

    @Override
    public Operation getNextOperation() {
        return nextOperation;
    }

    @Override
    public final void setNextOperation(Operation nextOperation) {
        this.nextOperation = nextOperation;
    }

    @Override
    public final void setLayerIndex(int layerIndex) {
        this.layerIndex = layerIndex;
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
}
