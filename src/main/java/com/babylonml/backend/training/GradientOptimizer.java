package com.babylonml.backend.training;

import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;

public interface GradientOptimizer {
    void optimize(TrainingExecutionContext executionContext,
                  float[] matrix, int matrixOffset, int rows, int columns, float[] gradient,
                  int gradientOffset, float learningRate);

    IntIntImmutablePair[] getRequiredMemoryAllocations(int rows, int columns);
}
