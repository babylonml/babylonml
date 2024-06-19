package com.babylonml.backend.training.optimizer;

import com.babylonml.backend.training.execution.TrainingExecutionContext;
import org.jspecify.annotations.NonNull;

public interface GradientOptimizer {
    void optimize(TrainingExecutionContext executionContext,
                  float[] matrix, int matrixOffset, int[] shape, float[] gradient,
                  int gradientOffset, float learningRate);

    int @NonNull [][] calculateRequiredMemoryAllocations(int[] shape);
}
