package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;

public class CrossEntropyByRowsFunction extends AbstractOperation {
    private final float[] expectedValues;

    private final int rows;
    private final int columns;

    public CrossEntropyByRowsFunction(int rows, int columns, float[] expectedValues,
                                      TrainingExecutionContext executionContext, Operation leftOperation) {
        super(executionContext, leftOperation, null);
        this.expectedValues = expectedValues;

        this.rows = rows;
        this.columns = columns;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public float[] getExpectedValues() {
        return expectedValues;
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
}
