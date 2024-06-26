package com.babylonml.backend.training.execution;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.operations.*;
import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;

import java.util.Collections;
import java.util.List;
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

            var dataShape = data.shape;
            this.batchWidth = TensorOperations.stride(dataShape) / dataShape.getInt(0);
        }

        @Override
        public int getSamplesCount() {
            return data.shape.getInt(0);
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
        public @NonNull IntImmutableList getMaxResultShape() {
            var dataShape = data.shape;
            var result = new int[dataShape.size()];
            result[0] = miniBatchSize;

            dataShape.getElements(1, result, 1, dataShape.size() - 1);

            return IntImmutableList.of(result);
        }

        @Override
        public @NonNull TensorPointer forwardPassCalculation() {
            var dataShape = data.shape;
            var currentBatchSnippedIndex = localMiniBatchIndex * miniBatchSize;
            var currentBatchSize = Math.min(miniBatchSize, dataShape.getInt(0) - currentBatchSnippedIndex);

            var shape = new int[dataShape.size()];
            shape[0] = currentBatchSize;

            dataShape.getElements(1, shape, 1, dataShape.size() - 1);
            var result = getExecutionContext().allocateForwardMemory(this, IntImmutableList.of(shape));

            var resultBuffer = result.buffer();
            var resultOffset = result.offset();


            var startIndex = currentBatchSnippedIndex * batchWidth;
            var width = currentBatchSize * batchWidth;

            System.arraycopy(data.data, startIndex, resultBuffer, resultOffset, width);

            return result;
        }

        @Override
        public int gitLocalMiniBatchIndex() {
            return localMiniBatchIndex;
        }

        @Override
        public @NonNull List<IntImmutableList> getForwardMemoryAllocations() {
            var dataShape = data.shape;
            var result = new int[dataShape.size()];
            result[0] = miniBatchSize;

            dataShape.getElements(1, result, 1, dataShape.size() - 1);

            return List.of(IntImmutableList.of(result));
        }

        @Override
        public @NonNull List<IntImmutableList> getBackwardMemoryAllocations() {
            return List.of();
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
        public void prepareForNextPropagation() {
            super.prepareForNextPropagation();

            globalMiniBatchIndex++;
            localMiniBatchIndex++;

            if (localMiniBatchIndex * miniBatchSize >= data.shape.getInt(0)) {
                localMiniBatchIndex = 0;
            }

            notifyListeners();
        }

        private void notifyListeners() {
            for (var listener : miniBatchListeners) {
                listener.onMiniBatchStart(globalMiniBatchIndex, Math.min(miniBatchSize,
                        data.shape.getInt(0) - localMiniBatchIndex * miniBatchSize));
            }
        }

        @Override
        public void startEpochExecution() {
            super.startEpochExecution();

            //will be set to 0 in prepareForNextPropagation
            localMiniBatchIndex = -1;
        }

        @Override
        public boolean getRequiresBackwardDerivativeChainValue() {
            return false;
        }
    }


}
