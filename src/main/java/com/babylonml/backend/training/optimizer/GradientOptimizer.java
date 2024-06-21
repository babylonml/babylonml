package com.babylonml.backend.training.optimizer;

import com.babylonml.backend.training.execution.TrainingExecutionContext;
import com.babylonml.backend.training.operations.Operation;
import org.jspecify.annotations.NonNull;

public interface GradientOptimizer {
    void optimize(TrainingExecutionContext executionContext,
                  float[] matrix, int matrixOffset, int[] shape, float[] gradient,
                  int gradientOffset, float learningRate, Operation operation);

    int @NonNull [][] calculateRequiredMemoryAllocations(int[] shape);
}
