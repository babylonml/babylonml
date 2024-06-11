package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;

public class SoftMaxByRows extends AbstractOperation {
    private final int maxRows;
    private final int maxColumns;

    public SoftMaxByRows(TrainingExecutionContext executionContext, Operation leftOperation) {
        super(executionContext, leftOperation, null);

        this.maxRows = leftOperation.getResultMaxRows();
        this.maxColumns = leftOperation.getResultMaxColumns();
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
    public int getResultMaxRows() {
        return maxRows;
    }

    @Override
    public int getResultMaxColumns() {
        return maxColumns;
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
