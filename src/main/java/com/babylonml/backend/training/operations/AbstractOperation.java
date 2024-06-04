package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;

public abstract class AbstractOperation implements Operation {
    protected Operation leftOperation;
    protected Operation rightOperation;

    protected final TrainingExecutionContext executionContext;

    protected long derivativeChainValue;

    protected Operation nextOperation;
    private int layerIndex = -1;

    public AbstractOperation(TrainingExecutionContext executionContext,
                             Operation leftOperation, Operation rightOperation) {

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
        this.derivativeChainValue = backwardDerivativeChainValue;
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
}
