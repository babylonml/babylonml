package com.babylonml.backend.training.optimizer;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import com.babylonml.backend.training.execution.InputSource;
import com.babylonml.backend.training.operations.MiniBatchListener;
import com.babylonml.backend.cpu.VectorOperations;
import org.jspecify.annotations.NonNull;

public class SimpleGradientDescentOptimizer implements GradientOptimizer, MiniBatchListener {
    private int scaleValue = 1;

    public SimpleGradientDescentOptimizer(@NonNull InputSource inputSource) {
        inputSource.addMiniBatchListener(this);
    }

    @Override
    public void optimize(@NonNull TrainingExecutionContext executionContext, float @NonNull [] matrix, int matrixOffset,
                         int[] shape, float @NonNull [] gradient, int gradientOffset, float learningRate) {
        var pointer = executionContext.allocateBackwardMemory(shape);
        var buffer = pointer.buffer();
        var bufferOffset = pointer.offset();

        var stride = TensorOperations.stride(shape);
        VectorOperations.multiplyVectorToScalar(gradient, gradientOffset,
                -learningRate / scaleValue, buffer, bufferOffset,
                stride);
        VectorOperations.addVectorToVector(matrix, matrixOffset, buffer, bufferOffset, matrix, matrixOffset,
                stride);
    }

    @Override
    public int @NonNull [][] calculateRequiredMemoryAllocations(int[] shape) {
        return new int [][]{
                shape
        };
    }

    @Override
    public void onMiniBatchStart(long miniBatchIndex, int miniBatchSize) {
        assert miniBatchSize > 0;
        scaleValue = miniBatchSize;
    }
}
