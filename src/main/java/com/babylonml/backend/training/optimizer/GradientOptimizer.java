package com.babylonml.backend.training.optimizer;

import com.babylonml.backend.training.execution.TrainingExecutionContext;
import com.babylonml.backend.training.operations.Operation;
import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;

import java.util.List;

public interface GradientOptimizer {
    void optimize(TrainingExecutionContext executionContext,
                  float[] matrix, int matrixOffset, IntImmutableList shape, float[] gradient,
                  int gradientOffset, float learningRate, Operation operation);

     @NonNull
     List<IntImmutableList> calculateRequiredMemoryAllocations(IntImmutableList shape);
}
