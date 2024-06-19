package com.babylonml.backend.training.optimizer;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import com.babylonml.backend.training.execution.InputSource;
import com.babylonml.backend.training.operations.MiniBatchListener;
import com.babylonml.backend.cpu.VectorOperations;

import org.jspecify.annotations.NonNull;

public class AdamOptimizer implements GradientOptimizer, MiniBatchListener {
    public static final float DEFAULT_BETA1 = 0.9f;
    public static final float DEFAULT_BETA2 = 0.999f;
    public static final float DEFAULT_EPSILON = 1e-8f;

    private float[] avgMovement;
    private float[] avgMovementSqr;


    private final float beta1;
    private final float beta2;
    private final float epsilon;

    private long batchIndex;
    private int scaleValue = 1;

    public AdamOptimizer(InputSource inputSource) {
        this(DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON, inputSource);
    }

    public AdamOptimizer(float beta1, float beta2, float epsilon, InputSource inputSource) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;

        inputSource.addMiniBatchListener(this);
    }

    @Override
    public void optimize(TrainingExecutionContext executionContext, float[] matrix, int matrixOffset,
                         int[] shape, float[] gradient, int gradientOffset, float learningRate) {
        int stride = TensorOperations.stride(shape);

        var avgMovementPointer = executionContext.allocateBackwardMemory(shape);
        var avgMovementBuffer = executionContext.getMemoryBuffer(avgMovementPointer.pointer());
        var avgMovementBufferOffset = TrainingExecutionContext.addressOffset(avgMovementPointer.pointer());


        var avgMovementSqrPointer = executionContext.allocateBackwardMemory(shape);
        var avgMovementSqrBuffer = executionContext.getMemoryBuffer(avgMovementSqrPointer.pointer());
        var avgMovementSqrBufferOffset = TrainingExecutionContext.addressOffset(avgMovementSqrPointer.pointer());

        updateAvgMovement(gradient, gradientOffset, avgMovement, avgMovementSqr,
                avgMovementBuffer, avgMovementBufferOffset, stride, beta1, beta2, scaleValue);
        movingAverageBiasCorrection(avgMovement, beta1, batchIndex, avgMovementBuffer, avgMovementBufferOffset,
                stride);
        movingAverageBiasCorrection(avgMovementSqr, beta2, batchIndex, avgMovementSqrBuffer, avgMovementSqrBufferOffset,
                stride);

        calculateCorrections(avgMovementBuffer, avgMovementBufferOffset, avgMovementSqrBuffer,
                avgMovementSqrBufferOffset, stride, learningRate, epsilon);
        VectorOperations.addVectorToVector(matrix, matrixOffset, avgMovementBuffer, avgMovementBufferOffset, matrix,
                matrixOffset, stride);

    }

    @Override
    public int @NonNull [][] calculateRequiredMemoryAllocations(int[] shape) {
        var stride = TensorOperations.stride(shape);
        this.avgMovement = new float[stride];
        this.avgMovementSqr = new float[stride];

        return new int[][]{
                shape, shape
        };
    }

    public static void updateAvgMovement(float[] gradient, int gradientOffset, float[] avgMovement,
                                         float[] avgMovementSqr, float[] avgMovementBuffer,
                                         int avgMovementBufferOffset, int size, float beta1, float beta2, int scale) {
        //g = g / scale
        //w[n] = b1 * w[n-1] + (1-b1) * g
        VectorOperations.multiplyVectorToScalar(avgMovement, 0, beta1,
                avgMovement, 0,
                size);
        VectorOperations.multiplyVectorToScalar(gradient, gradientOffset,
                (1 - beta1) / scale, avgMovementBuffer, avgMovementBufferOffset,
                size);
        VectorOperations.addVectorToVector(avgMovement, 0, avgMovementBuffer, avgMovementBufferOffset,
                avgMovement, 0, size);
        //v[n] = b2 * v[n-1] + (1-b2) * g^2
        VectorOperations.multiplyVectorToScalar(avgMovementSqr,
                0, beta2, avgMovementSqr, 0, size);
        VectorOperations.vectorToVectorElementWiseMultiplication(gradient, gradientOffset, gradient,
                gradientOffset, avgMovementBuffer, avgMovementBufferOffset, size);
        VectorOperations.multiplyVectorToScalar(avgMovementBuffer, avgMovementBufferOffset,
                (1 - beta2) / (scale * scale),
                avgMovementBuffer, avgMovementBufferOffset, size);
        VectorOperations.addVectorToVector(avgMovementSqr, 0, avgMovementBuffer, avgMovementBufferOffset,
                avgMovementSqr, 0, size);
    }

    private static void movingAverageBiasCorrection(float[] movingAverage,
                                                    float betta, long iteration, float[] result, int resultOffset, int size) {
        var coefficient = (float) (1.0 / (1.0 - Math.pow(betta, iteration)));
        VectorOperations.multiplyVectorToScalar(movingAverage, 0, coefficient, result, resultOffset,
                size);
    }

    public static void calculateCorrections(float[] correctedMovingAverage, int correctedMovingAverageOffset,
                                            float[] correctedMovingAverageSqr, int correctedMovingAverageSqrOffset,
                                            int size,
                                            float learningRate, float epsilon) {
        VectorOperations.vectorElementsSqrt(correctedMovingAverageSqr, correctedMovingAverageSqrOffset,
                correctedMovingAverageSqr, correctedMovingAverageSqrOffset,
                size);
        VectorOperations.addScalarToVector(epsilon, correctedMovingAverageSqr, correctedMovingAverageSqrOffset,
                correctedMovingAverageSqr, correctedMovingAverageSqrOffset,
                size);
        VectorOperations.divideScalarOnVectorElements(-learningRate, correctedMovingAverageSqr, correctedMovingAverageSqrOffset,
                correctedMovingAverageSqr, correctedMovingAverageSqrOffset, size);
        VectorOperations.vectorToVectorElementWiseMultiplication(correctedMovingAverage, correctedMovingAverageOffset,
                correctedMovingAverageSqr, correctedMovingAverageSqrOffset, correctedMovingAverage,
                correctedMovingAverageOffset,
                size);
    }


    @Override
    public void onMiniBatchStart(long miniBatchIndex, int miniBatchSize) {
        assert miniBatchSize > 0;

        batchIndex = miniBatchIndex;
        scaleValue = miniBatchSize;
    }
}
