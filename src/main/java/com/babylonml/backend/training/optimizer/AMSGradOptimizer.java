package com.babylonml.backend.training.optimizer;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.babylonml.backend.training.operations.InputSource;
import com.babylonml.backend.training.operations.MiniBatchListener;
import com.tornadoml.cpu.VectorOperations;

import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;

public class AMSGradOptimizer implements GradientOptimizer, MiniBatchListener {
    public static final float DEFAULT_BETA1 = 0.9f;
    public static final float DEFAULT_BETA2 = 0.999f;
    public static final float DEFAULT_EPSILON = 1e-8f;

    private float[] avgMovement;
    private float[] avgMovementSqr;
    private float[] correctedAvgMovementSqr;

    private final float beta1;
    private final float beta2;
    private final float epsilon;

    private int scaleValue = 1;

    public AMSGradOptimizer(InputSource inputSource) {
        this(DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON, inputSource);
    }

    public AMSGradOptimizer(float beta1, float beta2, float epsilon, InputSource inputSource) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;

        inputSource.addMiniBatchListener(this);
    }

    @Override
    public void onMiniBatchStart(long miniBatchIndex, int miniBatchSize) {
        scaleValue = miniBatchSize;
    }

    @Override
    public void optimize(TrainingExecutionContext executionContext, float[] matrix, int matrixOffset,
                         int rows, int columns, float[] gradient, int gradientOffset, float learningRate) {
        final int size = rows * columns;

        var calculationBufferPointer = executionContext.allocateBackwardMemory(rows, columns);
        var calculationBuffer = executionContext.getMemoryBuffer(calculationBufferPointer);
        var calculationBufferOffset = TrainingExecutionContext.addressOffset(calculationBufferPointer);

        AdamOptimizer.updateAvgMovement(gradient, gradientOffset, avgMovement, avgMovementSqr, calculationBuffer,
                calculationBufferOffset, size, beta1, beta2, scaleValue);
        correctAvgMovementSqr(avgMovementSqr, 0, correctedAvgMovementSqr,
                0, size);
        calculateCorrections(avgMovement, 0, correctedAvgMovementSqr, 0,
                calculationBuffer, calculationBufferOffset, size, learningRate, epsilon);
        VectorOperations.addVectorToVector(matrix, 0, calculationBuffer, calculationBufferOffset, matrix, 0,
                size);
    }

    @Override
    public IntIntImmutablePair[] calculateRequiredMemoryAllocations(int rows, int columns) {
        avgMovement = new float[rows * columns];
        avgMovementSqr = new float[rows * columns];
        correctedAvgMovementSqr = new float[rows * columns];

        return new IntIntImmutablePair[] {
                new IntIntImmutablePair(rows, columns),
        };
    }

    @SuppressWarnings("SameParameterValue")
    private static void correctAvgMovementSqr(float[] avgMovementSqr, int avgMovementSqrOffset,
                                              float[] correctedAvgMovementSqr, int correctedAvgMovementSqrOffset, int size) {
        VectorOperations.maxBetweenVectorElements(avgMovementSqr, avgMovementSqrOffset, correctedAvgMovementSqr,
                correctedAvgMovementSqrOffset,
                correctedAvgMovementSqr, correctedAvgMovementSqrOffset, size);
    }

    @SuppressWarnings("SameParameterValue")
    private static void calculateCorrections(float[] movingAverage, int movingAverageOffset,
                                             float[] correctedMovingAverageSqr, int correctedMovingAverageSqrOffset,
                                             float[] calculationBuffer, int calculationBufferOffset, int size,
                                             float learningRate, float epsilon) {
        VectorOperations.vectorElementsSqrt(correctedMovingAverageSqr, correctedMovingAverageSqrOffset,
                calculationBuffer, calculationBufferOffset, size);
        VectorOperations.addScalarToVector(epsilon, calculationBuffer, calculationBufferOffset, calculationBuffer,
                calculationBufferOffset, size);
        VectorOperations.divideScalarOnVectorElements(-learningRate, calculationBuffer, calculationBufferOffset,
                calculationBuffer, calculationBufferOffset, size);
        VectorOperations.vectorToVectorElementWiseMultiplication(movingAverage, movingAverageOffset,
                calculationBuffer, calculationBufferOffset,
                calculationBuffer, calculationBufferOffset, size);
    }
}
