package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;
import org.jspecify.annotations.NonNull;

import java.util.Collections;
import java.util.Set;
import java.util.WeakHashMap;

public class MiniBatchInputSource extends AbstractOperation implements StartOperation, InputSource {
    private final float[][] data;

    private final int columns;
    private final int miniBatchSize;

    private int currentRowIndex;
    private long miniBatchIndex = 0;

    private final Set<MiniBatchListener> miniBatchListeners = Collections.newSetFromMap(new WeakHashMap<>());

    public MiniBatchInputSource(float[][] data, int columns, int miniBatchSize,
                                TrainingExecutionContext executionContext) {
        super(executionContext, null, null);

        this.data = data;
        this.columns = columns;
        this.miniBatchSize = miniBatchSize;
        this.currentRowIndex = data.length;
    }

    @Override
    public int getResultMaxRows() {
        return miniBatchSize;
    }

    @Override
    public int getResultMaxColumns() {
        return columns;
    }

    @Override
    public long forwardPassCalculation() {
        var currentBatchSize = Math.min(miniBatchSize, data.length - currentRowIndex);
        var result = executionContext.allocateForwardMemory(currentBatchSize, columns);

        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        for (int i = 0; i < currentBatchSize; i++) {
            var dataRow = data[currentRowIndex + i];
            assert dataRow.length == columns;

            System.arraycopy(dataRow, 0, resultBuffer, resultOffset + i * columns, columns);
        }

        return result;
    }

    @Override
    public IntIntImmutablePair[] getForwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(miniBatchSize, columns)
        };
    }

    @Override
    public IntIntImmutablePair[] getBackwardMemoryAllocations() {
        return new IntIntImmutablePair[0];
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
    public boolean requiresBackwardDerivativeChainValue() {
        return false;
    }

    @Override
    public void prepareForNextPropagation() {
        super.prepareForNextPropagation();

        currentRowIndex = currentRowIndex + miniBatchSize;
        miniBatchIndex++;

        if (currentRowIndex >= data.length) {
            currentRowIndex = 0;
        }

        for (var listener : miniBatchListeners) {
            listener.onMiniBatchStart(miniBatchIndex, Math.min(miniBatchSize, data.length - currentRowIndex));
        }
    }

    @Override
    public void addMiniBatchListener(@NonNull  MiniBatchListener listener) {
        miniBatchListeners.add(listener);
    }

    @Override
    public void calculateGradientUpdate() {
        // No gradient update required
    }
}
