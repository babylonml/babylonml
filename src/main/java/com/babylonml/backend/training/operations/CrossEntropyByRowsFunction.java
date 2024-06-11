package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;

public class CrossEntropyByRowsFunction extends AbstractOperation {

    private final int maxRows;
    private final int maxColumns;

    public CrossEntropyByRowsFunction(Operation expectedValues,
                                      TrainingExecutionContext executionContext, Operation leftOperation) {
        super(executionContext, leftOperation, expectedValues);

        this.maxRows = leftOperation.getResultMaxRows();
        this.maxColumns = leftOperation.getResultMaxColumns();
    }

    public int getResultMaxRows() {
        return maxRows;
    }

    public int getResultMaxColumns() {
        return maxColumns;
    }

    public Operation getExpectedValues() {
        return rightOperation;
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
    public IntIntImmutablePair[] getForwardMemoryAllocations() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in forward pass");
    }

    @Override
    public IntIntImmutablePair[] getBackwardMemoryAllocations() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in backward pass");
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return false;
    }
}
