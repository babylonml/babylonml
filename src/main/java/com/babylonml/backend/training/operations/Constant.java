package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;

@SuppressWarnings("unused")
public final class Constant extends AbstractOperation {
    private final float[] constant;

    private final int rows;
    private final int columns;

    public Constant(TrainingExecutionContext executionContext, float[] constant, int rows, int columns) {
        super(executionContext, null, null);
        this.constant = constant;
        this.rows = rows;
        this.columns = columns;
    }


    @Override
    public long forwardPassCalculation() {
        var result = executionContext.allocateForwardMemory(rows * columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        System.arraycopy(constant, 0, resultBuffer, resultOffset, rows * columns);

        return result;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public int getForwardMemorySize() {
        return rows * columns;
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
