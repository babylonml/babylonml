package com.babylonml.backend.training.optimizer;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.babylonml.backend.training.operations.InputSource;
import com.babylonml.backend.training.operations.MiniBatchListener;
import com.tornadoml.cpu.VectorOperations;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;
import org.jspecify.annotations.NonNull;

public class SimpleGradientDescentOptimizer implements GradientOptimizer, MiniBatchListener {
    private int scaleValue = 1;

    public SimpleGradientDescentOptimizer(@NonNull InputSource inputSource) {
        inputSource.addMiniBatchListener(this);
    }

    @Override
    public void optimize(@NonNull TrainingExecutionContext executionContext, float @NonNull [] matrix, int matrixOffset,
                         int rows, int columns, float @NonNull [] gradient, int gradientOffset, float learningRate) {
        var address = executionContext.allocateBackwardMemory(rows, columns);
        var buffer = executionContext.getMemoryBuffer(address);
        var bufferOffset = TrainingExecutionContext.addressOffset(address);

        VectorOperations.multiplyVectorToScalar(gradient, gradientOffset,
                -learningRate / scaleValue, buffer, bufferOffset,
                rows * columns);
        VectorOperations.addVectorToVector(matrix, matrixOffset, buffer, bufferOffset, matrix, matrixOffset,
                rows * columns);
    }

    @Override
    public IntIntImmutablePair[] calculateRequiredMemoryAllocations(int rows, int columns) {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(rows, columns)
        };
    }

    @Override
    public void onMiniBatchStart(long miniBatchIndex, int miniBatchSize) {
        assert miniBatchSize > 0;
        scaleValue = miniBatchSize;
    }
}
