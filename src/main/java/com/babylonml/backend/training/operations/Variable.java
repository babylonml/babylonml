package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.babylonml.backend.training.GradientOptimizer;

public final class Variable extends AbstractOperation {
    private final GradientOptimizer optimizer;
    private final float learningRate;

    private final float[] variable;

    private final int rows;
    private final int columns;

    public Variable(TrainingExecutionContext executionContext, GradientOptimizer optimizer, float[] variable, int rows,
                    int columns, float learningRate) {
        super(executionContext, null, null);

        this.optimizer = optimizer;
        this.variable = variable;
        this.rows = rows;
        this.columns = columns;
        this.learningRate = learningRate;

        executionContext.registerOperation(this);
    }

    @Override
    public long forwardPassCalculation() {
        var result = executionContext.allocateForwardMemory(rows * columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        System.arraycopy(variable, 0, resultBuffer, resultOffset, rows * columns);
        return result;
    }


    @Override
    public int getForwardMemorySize() {
        return rows * columns;
    }

    @Override
    public void updateBackwardDerivativeChainValue(long backwardDerivativeChainValue) {
        super.updateBackwardDerivativeChainValue(backwardDerivativeChainValue);

        var derivativeBuffer = executionContext.getMemoryBuffer(backwardDerivativeChainValue);
        var derivativeOffset = TrainingExecutionContext.addressOffset(backwardDerivativeChainValue);

        optimizer.optimize(executionContext, variable, 0, rows, columns, derivativeBuffer,
                derivativeOffset, learningRate);
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
}
