package com.babylonml.backend.training;

import com.tornadoml.cpu.VectorOperations;

public class SimpleGradientDescentOptimizer implements GradientOptimizer {
    private final int scale;

    public SimpleGradientDescentOptimizer(int scale) {
        this.scale = scale;
    }

    @Override
    public void optimize(TrainingExecutionContext executionContext, float[] matrix, int matrixOffset,
                         int rows, int columns, float[] gradient, int gradientOffset, float learningRate) {
        var address = executionContext.allocateBackwardMemory(rows * columns);
        var buffer = executionContext.getMemoryBuffer(address);
        var bufferOffset = TrainingExecutionContext.addressOffset(address);

        VectorOperations.multiplyVectorToScalar(gradient, gradientOffset,
                -learningRate / scale, buffer, bufferOffset,
                rows * columns);
        VectorOperations.addVectorToVector(matrix, matrixOffset, buffer, bufferOffset, matrix, matrixOffset,
                rows * columns);
    }

    @Override
    public int getRequiredMemorySize(int rows, int columns) {
        return rows * columns;
    }
}
