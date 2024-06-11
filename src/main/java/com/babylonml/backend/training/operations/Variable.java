package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.babylonml.backend.training.GradientOptimizer;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

public final class Variable extends AbstractOperation implements StartOperation {
    @NonNull
    private final GradientOptimizer optimizer;
    private final float learningRate;

    private final float @NonNull [] data;

    private final int rows;
    private final int columns;

    public Variable(@NonNull TrainingExecutionContext executionContext, @NonNull GradientOptimizer optimizer,
                    float @NonNull [] data, int rows, int columns, float learningRate) {
        this(null, executionContext, optimizer, data, rows, columns, learningRate);
    }

    public Variable(@Nullable String name, @NonNull TrainingExecutionContext executionContext,
                    @NonNull GradientOptimizer optimizer,
                    float @NonNull [] data, int rows, int columns, float learningRate) {
        super(name, executionContext, null, null);

        this.optimizer = optimizer;
        this.data = data;
        this.rows = rows;
        this.columns = columns;
        this.learningRate = learningRate;
    }

    @Override
    public int getResultMaxRows() {
        return rows;
    }

    @Override
    public int getResultMaxColumns() {
        return columns;
    }

    @Override
    public long forwardPassCalculation() {
        var result = executionContext.allocateForwardMemory(rows, columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        System.arraycopy(data, 0, resultBuffer, resultOffset, rows * columns);
        return result;
    }


    @Override
    public IntIntImmutablePair[] getForwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(rows, columns)
        };
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
    public IntIntImmutablePair[] getBackwardMemoryAllocations() {
        return optimizer.getRequiredMemoryAllocations(rows, columns);
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return true;
    }

    public float[] getData() {
        return data;
    }

    @Override
    public void calculateGradientUpdate() {
        var derivativeBuffer = executionContext.getMemoryBuffer(derivativeChainPointer);
        var derivativeOffset = TrainingExecutionContext.addressOffset(derivativeChainPointer);

        optimizer.optimize(executionContext, data, 0, rows, columns, derivativeBuffer,
                derivativeOffset, learningRate);
    }
}
