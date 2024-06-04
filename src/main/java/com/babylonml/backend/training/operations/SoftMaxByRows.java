package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;

public class SoftMaxByRows extends AbstractOperation {
    private final int rows;
    private final int columns;

    public SoftMaxByRows(TrainingExecutionContext executionContext, Operation leftOperation, int rows, int columns) {
        super(executionContext, leftOperation, null);

        this.rows = rows;
        this.columns = columns;
    }

    @Override
    public long forwardPassCalculation() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in forward pass");
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in backward pass");
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in backward pass");
    }

    @Override
    public int getForwardMemorySize() {
        return 0;
    }

    @Override
    public int getBackwardMemorySize() {
        return 0;
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return false;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }
}
