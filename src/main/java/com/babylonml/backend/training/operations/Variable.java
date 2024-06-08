package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.babylonml.backend.training.GradientOptimizer;

public final class Variable extends AbstractOperation implements StartOperation {
    private final GradientOptimizer optimizer;
    private final float learningRate;

    private final float[] data;

    private final int rows;
    private final int columns;

    public Variable(TrainingExecutionContext executionContext, GradientOptimizer optimizer, float[] data, int rows,
                    int columns, float learningRate) {
        super(executionContext, null, null);

        this.optimizer = optimizer;
        this.data = data;
        this.rows = rows;
        this.columns = columns;
        this.learningRate = learningRate;
    }

    @Override
    public long forwardPassCalculation() {
        var result = executionContext.allocateForwardMemory(rows * columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        System.arraycopy(data, 0, resultBuffer, resultOffset, rows * columns);
        return result;
    }


    @Override
    public int getForwardMemorySize() {
        return rows * columns;
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
    public int getBackwardMemorySize() {
        return optimizer.getRequiredMemorySize(rows, columns);
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return true;
    }

    public float[] getData() {
        return data;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    @Override
    public void calculateGradientUpdate() {
        var derivativeBuffer = executionContext.getMemoryBuffer(derivativeChainValue);
        var derivativeOffset = TrainingExecutionContext.addressOffset(derivativeChainValue);

        optimizer.optimize(executionContext, data, 0, rows, columns, derivativeBuffer,
                derivativeOffset, learningRate);
    }
}
