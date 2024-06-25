package com.babylonml.backend.training.optimizer;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import com.babylonml.backend.training.execution.ContextInputSource;
import com.babylonml.backend.training.operations.MiniBatchListener;
import com.babylonml.backend.cpu.VectorOperations;
import com.babylonml.backend.training.operations.Operation;
import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;

import java.util.List;

public class SimpleGradientDescentOptimizer implements GradientOptimizer, MiniBatchListener {
    private int scaleValue = 1;

    public SimpleGradientDescentOptimizer(@NonNull ContextInputSource inputSource) {
        inputSource.addMiniBatchListener(this);
    }

    @Override
    public void optimize(@NonNull TrainingExecutionContext executionContext, float @NonNull [] matrix, int matrixOffset,
                         IntImmutableList shape, float @NonNull [] gradient, int gradientOffset, float learningRate, Operation operation) {
        var pointer = executionContext.allocateBackwardMemory(operation, shape);
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
    public @NonNull List<IntImmutableList> calculateRequiredMemoryAllocations(IntImmutableList shape) {
        return List.of(shape);
    }

    @Override
    public void onMiniBatchStart(long miniBatchIndex, int miniBatchSize) {
        assert miniBatchSize > 0;
        scaleValue = miniBatchSize;
    }
}
