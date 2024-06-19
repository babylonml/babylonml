package com.babylonml.backend.training.execution;

import com.babylonml.backend.training.operations.AbstractOperation;
import com.babylonml.backend.training.operations.MiniBatchListener;
import com.babylonml.backend.training.operations.StartOperation;
import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;

import java.util.Collections;
import java.util.Set;
import java.util.WeakHashMap;

class MiniBatchVectorInputSource extends AbstractOperation implements StartOperation, InputSource {
    private final float[][] data;

    private final int columns;
    private final int miniBatchSize;

    private int currentRowIndex;
    private long miniBatchIndex = 0;

    private final Set<MiniBatchListener> miniBatchListeners = Collections.newSetFromMap(new WeakHashMap<>());

    MiniBatchVectorInputSource(float[][] data, int columns, int miniBatchSize,
                               TrainingExecutionContext executionContext) {
        super(executionContext, null, null);

        this.data = data;
        this.columns = columns;
        this.miniBatchSize = miniBatchSize;
        this.currentRowIndex = data.length;
    }

    @Override
    public int @NonNull [] getMaxResultShape() {
        return new int[]{
                miniBatchSize, columns
        };
    }

    @Override
    public @NonNull TensorPointer forwardPassCalculation() {
        var currentBatchSize = Math.min(miniBatchSize, data.length - currentRowIndex);
        var result = executionContext.allocateForwardMemory(currentBatchSize, columns);

        var resultBuffer = result.buffer();
        var resultOffset = result.offset();

        for (int i = 0; i < currentBatchSize; i++) {
            var dataRow = data[currentRowIndex + i];
            assert dataRow.length == columns;

            System.arraycopy(dataRow, 0, resultBuffer, resultOffset + i * columns, columns);
        }

        return result;
    }

    @NotNull
    @Override
    public int @NonNull [][] getForwardMemoryAllocations() {
        return new int[][]{
                new int[]{miniBatchSize, columns},
        };
    }

    @Override
    public int @NonNull [][] getBackwardMemoryAllocations() {
        return new int[0][];
    }

    @Override
    public @NonNull TensorPointer leftBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public @NonNull TensorPointer rightBackwardDerivativeChainValue() {
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

        notifyListeners();
    }

    private void notifyListeners() {
        for (var listener : miniBatchListeners) {
            listener.onMiniBatchStart(miniBatchIndex, Math.min(miniBatchSize, data.length - currentRowIndex));
        }
    }

    @Override
    public void addMiniBatchListener(@NonNull MiniBatchListener listener) {
        miniBatchListeners.add(listener);
    }

    @Override
    public int getDataSize() {
        return data.length;
    }

    @Override
    public void calculateGradientUpdate() {
        // No gradient update required
    }

    @Override
    public void startEpochExecution() {
        super.startEpochExecution();

        currentRowIndex = 0;
        miniBatchIndex = 0;

        notifyListeners();
    }
}
