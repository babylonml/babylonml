package com.babylonml.backend.training.optimizer

import com.babylonml.backend.training.execution.TrainingExecutionContext
import com.babylonml.backend.training.operations.Operation
import it.unimi.dsi.fastutil.ints.IntImmutableList

interface GradientOptimizer {
    fun optimize(
        executionContext: TrainingExecutionContext,
        matrix: FloatArray, matrixOffset: Int, shape: IntImmutableList, gradient: FloatArray,
        gradientOffset: Int, learningRate: Float, operation: Operation
    )

    fun calculateRequiredMemoryAllocations(shape: IntImmutableList): List<IntImmutableList>
}
