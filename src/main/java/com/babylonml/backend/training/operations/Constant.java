package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;

import java.util.Collections;
import java.util.Set;
import java.util.WeakHashMap;

@SuppressWarnings("unused")
public final class Constant extends AbstractOperation implements StartOperation, InputSource {
    private final float[] constant;

    private final int rows;
    private final int columns;
    private long miniBatchIndex;

    private final Set<MiniBatchListener> miniBatchListeners = Collections.newSetFromMap(new WeakHashMap<>());

    public Constant(TrainingExecutionContext executionContext, float[] constant, int rows, int columns) {
        this(null, executionContext, constant, rows, columns);
    }

    public Constant(String name, TrainingExecutionContext executionContext, float[] constant, int rows, int columns) {
        super(name, executionContext, null, null);
        this.constant = constant;
        this.rows = rows;
        this.columns = columns;
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
    public IntIntImmutablePair[] getForwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(rows, columns)
        };
    }

    @Override
    public IntIntImmutablePair[] getBackwardMemoryAllocations() {
        return new IntIntImmutablePair[0];
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return false;
    }

    @Override
    public void calculateGradientUpdate() {
        // No gradient update required
    }

    @Override
    public void addMiniBatchListener(MiniBatchListener listener) {
        miniBatchListeners.add(listener);
    }

    @Override
    public void prepareForNextPropagation() {
        super.prepareForNextPropagation();

        miniBatchIndex++;
        for (var listener : miniBatchListeners) {
            listener.onMiniBatchStart(miniBatchIndex, rows);
        }
    }
}
