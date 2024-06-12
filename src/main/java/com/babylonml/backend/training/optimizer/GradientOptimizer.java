package com.babylonml.backend.training.optimizer;

import com.babylonml.backend.training.TrainingExecutionContext;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;

public interface GradientOptimizer {
    void optimize(TrainingExecutionContext executionContext,
                  float[] matrix, int matrixOffset, int rows, int columns, float[] gradient,
                  int gradientOffset, float learningRate);

    IntIntImmutablePair[] calculateRequiredMemoryAllocations(int rows, int columns);
}
