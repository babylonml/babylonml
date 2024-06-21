package com.babylonml.backend.training.execution;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.operations.*;
import org.jspecify.annotations.NonNull;

import java.util.Collections;
import java.util.Set;
import java.util.WeakHashMap;

public class MiniBatchTensorInputDataSource implements UserInputSource {
    private final Tensor data;

    public MiniBatchTensorInputDataSource(Tensor data) {
        this.data = data;
    }

    @Override
    public ContextInputSource convertToContextInputSource(int miniBatchSize,
                                                          TrainingExecutionContext executionContext) {
        return new MiniBatchTensorInputDataSourceOperation(data, miniBatchSize, executionContext);
    }

    private final class MiniBatchTensorInputDataSourceOperation extends AbstractOperation implements
            StartOperation, ContextInputSource {
        private long globalMiniBatchIndex = 0;
        private int localMiniBatchIndex = 0;

        private final int miniBatchSize;
        private final int batchWidth;

        private final Set<MiniBatchListener> miniBatchListeners = Collections.newSetFromMap(new WeakHashMap<>());

        private MiniBatchTensorInputDataSourceOperation(Tensor data, int miniBatchSize,
                                                        TrainingExecutionContext executionContext) {
            super(executionContext, null, null);

            if (data.isEmpty()) {
                throw new IllegalArgumentException("Data must have at least one entry.");
            }
            if (miniBatchSize <= 0) {
                throw new IllegalArgumentException("Mini batch size must be greater than 0.");
            }

            this.miniBatchSize = miniBatchSize;

            var dataShape = data.getShape();
            this.batchWidth = TensorOperations.stride(dataShape) / dataShape[0];
        }

        @Override
        public int getSamplesCount() {
            return data.getShape()[0];
        }

        @Override
        public int getMiniBatchSize() {
            return miniBatchSize;
        }

        @Override
        public void addMiniBatchListener(@NonNull MiniBatchListener listener) {
            miniBatchListeners.add(listener);
        }

        @Override
        public void calculateGradientUpdate() {
            //nothing to do
        }

        @Override
        public int @NonNull [] getMaxResultShape() {
            var dataShape = data.getShape();
            var result = new int[dataShape.length];
            result[0] = miniBatchSize;

            System.arraycopy(dataShape, 1, result, 1, dataShape.length - 1);

            return result;
        }

        @Override
        public @NonNull TensorPointer forwardPassCalculation() {
            var dataShape = data.getShape();
            var currentBatchSnippedIndex = localMiniBatchIndex * miniBatchSize;
            var currentBatchSize = Math.min(miniBatchSize, dataShape[0] - currentBatchSnippedIndex);

            var shape = new int[dataShape.length];
            shape[0] = currentBatchSize;

            System.arraycopy(dataShape, 1, shape, 1, dataShape.length - 1);
            var result = executionContext.allocateForwardMemory(this, shape);

            var resultBuffer = result.buffer();
            var resultOffset = result.offset();


            var startIndex = currentBatchSnippedIndex * batchWidth;
            var width = currentBatchSize * batchWidth;

            System.arraycopy(data.getData(), startIndex, resultBuffer, resultOffset, width);

            return result;
        }

        @Override
        public int gitLocalMiniBatchIndex() {
            return localMiniBatchIndex;
        }

        @Override
        public int @NonNull [][] getForwardMemoryAllocations() {
            var dataShape = data.getShape();
            var result = new int[dataShape.length];
            result[0] = miniBatchSize;

            System.arraycopy(dataShape, 1, result, 1, dataShape.length - 1);

            return new int[][]{result};
        }

        @Override
        public int @NonNull [][] getBackwardMemoryAllocations() {
            return new int[0][0];
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

            globalMiniBatchIndex++;
            localMiniBatchIndex++;

            if (localMiniBatchIndex * miniBatchSize >= data.getShape()[0]) {
                localMiniBatchIndex = 0;
            }

            notifyListeners();
        }

        private void notifyListeners() {
            for (var listener : miniBatchListeners) {
                listener.onMiniBatchStart(globalMiniBatchIndex, Math.min(miniBatchSize,
                        data.getShape()[0] - localMiniBatchIndex * miniBatchSize));
            }
        }

        @Override
        public void startEpochExecution() {
            super.startEpochExecution();

            //will be set to 0 in prepareForNextPropagation
            localMiniBatchIndex = -1;
        }
    }


}
