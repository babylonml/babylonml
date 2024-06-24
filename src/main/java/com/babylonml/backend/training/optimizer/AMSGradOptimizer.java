package com.babylonml.backend.training.optimizer;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import com.babylonml.backend.training.execution.ContextInputSource;
import com.babylonml.backend.training.operations.MiniBatchListener;
import com.babylonml.backend.cpu.VectorOperations;

import com.babylonml.backend.training.operations.Operation;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.Objects;

public class AMSGradOptimizer implements GradientOptimizer, MiniBatchListener {
    public static final float DEFAULT_BETA1 = 0.9f;
    public static final float DEFAULT_BETA2 = 0.999f;
    public static final float DEFAULT_EPSILON = 1e-8f;

    private float @Nullable [] avgMovement;
    private float @Nullable [] avgMovementSqr;
    private float @Nullable [] correctedAvgMovementSqr;

    private final float beta1;
    private final float beta2;
    private final float epsilon;

    private int scaleValue = 1;

    public AMSGradOptimizer(ContextInputSource inputSource) {
        this(DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON, inputSource);
    }

    public AMSGradOptimizer(float beta1, float beta2, float epsilon, ContextInputSource inputSource) {
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
                         int[] shape, float[] gradient, int gradientOffset, float learningRate, Operation operation) {
        Objects.requireNonNull(avgMovement);
        Objects.requireNonNull(avgMovementSqr);
        Objects.requireNonNull(correctedAvgMovementSqr);

        final int stride = TensorOperations.stride(shape);

        var calculationBufferPointer = executionContext.allocateBackwardMemory(operation, shape);
        var calculationBuffer = executionContext.getMemoryBuffer(calculationBufferPointer.pointer());
        var calculationBufferOffset = TrainingExecutionContext.addressOffset(calculationBufferPointer.pointer());

        AdamOptimizer.updateAvgMovement(gradient, gradientOffset, avgMovement, avgMovementSqr, calculationBuffer,
                calculationBufferOffset, stride, beta1, beta2, scaleValue);
        correctAvgMovementSqr(avgMovementSqr, 0, correctedAvgMovementSqr,
                0, stride);
        calculateCorrections(avgMovement, 0, correctedAvgMovementSqr, 0,
                calculationBuffer, calculationBufferOffset, stride, learningRate, epsilon);
        VectorOperations.addVectorToVector(matrix, 0, calculationBuffer, calculationBufferOffset, matrix, 0,
                stride);
    }

    @Override
    public int @NonNull [][] calculateRequiredMemoryAllocations(int[] shape) {
        var stride = TensorOperations.stride(shape);
        avgMovement = new float[stride];
        avgMovementSqr = new float[stride];
        correctedAvgMovementSqr = new float[stride];

        return new int[][]{
                shape
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
