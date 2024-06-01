package com.babylonml.backend.training;

public interface GradientOptimizer {
    void optimize(TrainingExecutionContext executionContext,
                  float[] matrix, int matrixOffset, int rows, int columns, float[] gradient,
                  int gradientOffset, float learningRate);

    int getRequiredMemorySize(int rows, int columns);
}
