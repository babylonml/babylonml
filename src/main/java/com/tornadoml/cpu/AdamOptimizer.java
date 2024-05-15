package com.tornadoml.cpu;

public final class AdamOptimizer implements WeightsOptimizer {
    private static final ThreadLocal<float[]> avgWeightsMovementBuffer = new ThreadLocal<>();
    private static final ThreadLocal<float[]> avgBiasesMovementBuffer = new ThreadLocal<>();
    private static final ThreadLocal<float[]> avgWeightsMovementSqrBuffer = new ThreadLocal<>();
    private static final ThreadLocal<float[]> avgBiasesMovementSqrBuffer = new ThreadLocal<>();

    private final float[] avgWeightsMovement;
    private final float[] avgBiasesMovement;
    private final float[] avgWeightsMovementSqr;
    private final float[] avgBiasesMovementSqr;

    private int iteration;

    public AdamOptimizer(int weightsSize, int biasesSize) {
        avgWeightsMovement = new float[weightsSize];
        avgBiasesMovement = new float[biasesSize];

        avgWeightsMovementSqr = new float[weightsSize];
        avgBiasesMovementSqr = new float[biasesSize];
    }

    @Override
    public void optimize(float[] weights, float[] weightsGradient, int weightsLength, float[] biases, float[] biasesGradient, int biasesLength, float learningRate) {
        iteration++;

        var avgWeightsMovementBuffer = getAvgWeightsMovementBuffer(weightsLength);

        var avgBiasesMovementBuffer = AdamOptimizer.avgBiasesMovementBuffer.get();
        if (avgBiasesMovementBuffer == null || avgBiasesMovementBuffer.length < biasesLength) {
            avgBiasesMovementBuffer = new float[biasesLength];
            AdamOptimizer.avgBiasesMovementBuffer.set(avgBiasesMovementBuffer);
        }

        var avgWeightsMovementSqrBuffer = AdamOptimizer.avgWeightsMovementSqrBuffer.get();
        if (avgWeightsMovementSqrBuffer == null || avgWeightsMovementSqrBuffer.length < weightsLength) {
            avgWeightsMovementSqrBuffer = new float[weightsLength];
            AdamOptimizer.avgWeightsMovementSqrBuffer.set(avgWeightsMovementSqrBuffer);
        }

        var avgBiasesMovementSqrBuffer = AdamOptimizer.avgBiasesMovementSqrBuffer.get();
        if (avgBiasesMovementSqrBuffer == null || avgBiasesMovementSqrBuffer.length < biasesLength) {
            avgBiasesMovementSqrBuffer = new float[biasesLength];
            AdamOptimizer.avgBiasesMovementSqrBuffer.set(avgBiasesMovementSqrBuffer);
        }


        updateAvgMovement(weightsGradient, avgWeightsMovement, avgWeightsMovementSqr, weightsLength, avgWeightsMovementBuffer);
        movingAverageBiasCorrection(avgWeightsMovement, 0.9f, iteration, avgWeightsMovementBuffer);
        movingAverageBiasCorrection(avgWeightsMovementSqr, 0.999f, iteration, avgWeightsMovementSqrBuffer);
        calculateCorrections(avgWeightsMovementBuffer, avgWeightsMovementSqrBuffer, learningRate);
        VectorOperations.addVectorToVector(weights, avgWeightsMovementBuffer, weights, weightsLength);

        updateAvgMovement(biasesGradient, avgBiasesMovement, avgBiasesMovementSqr, biasesLength, avgBiasesMovementBuffer);
        movingAverageBiasCorrection(avgBiasesMovement, 0.9f, iteration, avgBiasesMovementBuffer);
        movingAverageBiasCorrection(avgBiasesMovementSqr, 0.999f, iteration, avgBiasesMovementSqrBuffer);
        calculateCorrections(avgBiasesMovementBuffer, avgBiasesMovementSqrBuffer, learningRate);

        VectorOperations.addVectorToVector(biases, avgBiasesMovementBuffer, biases, biasesLength);
    }

    private static float[] getAvgWeightsMovementBuffer(int weightsLength) {
        var avgWeightsMovementBuffer = AdamOptimizer.avgWeightsMovementBuffer.get();
        if (avgWeightsMovementBuffer == null || avgWeightsMovementBuffer.length < weightsLength) {
            avgWeightsMovementBuffer = new float[weightsLength];
            AdamOptimizer.avgWeightsMovementBuffer.set(avgWeightsMovementBuffer);
        }

        return avgWeightsMovementBuffer;
    }

    public static void updateAvgMovement(float[] weightsDelta, float[] avgWeightsMovements, float[] avgWeightsMovementSqr,
                                         int weightsSize, float[] buffer) {
        //w[n] = 0.9 * w[n-1] + 0.1 * g
        VectorOperations.multiplyVectorToScalar(avgWeightsMovements, 0, 0.9f, avgWeightsMovements, 0,
                weightsSize);
        VectorOperations.multiplyVectorToScalar(weightsDelta, 0, 0.1f, buffer, 0,
                weightsSize);
        VectorOperations.addVectorToVector(avgWeightsMovements, buffer, avgWeightsMovements, weightsSize);

        //v[n] = 0.999 * v[n-1] + 0.001 * g^2
        VectorOperations.multiplyVectorToScalar(avgWeightsMovementSqr, 0, 0.999f, avgWeightsMovementSqr, 0,
                weightsSize);
        VectorOperations.vectorToVectorScalarMultiplication(weightsDelta, weightsDelta, buffer, weightsSize);
        VectorOperations.multiplyVectorToScalar(buffer, 0, 0.001f, buffer, 0, weightsSize);
        VectorOperations.addVectorToVector(avgWeightsMovementSqr, buffer, avgWeightsMovementSqr, weightsSize);
    }

    private static void movingAverageBiasCorrection(float[] movingAverage, float betta, int iteration, float[] result) {
        var coefficient = (float) (1.0 / (1.0 - Math.pow(betta, iteration)));
        VectorOperations.multiplyVectorToScalar(movingAverage, 0, coefficient, result, 0,
                movingAverage.length);
    }

    private static void calculateCorrections(float[] correctedMovingAverage, float[] correctedMovingAverageSqr, float learningRate) {
        VectorOperations.vectorElementsSqrt(correctedMovingAverageSqr, correctedMovingAverageSqr, correctedMovingAverageSqr.length);
        VectorOperations.addScalarToVector(1e-8f, correctedMovingAverageSqr, correctedMovingAverageSqr, correctedMovingAverageSqr.length);
        VectorOperations.divideScalarOnVectorElements(-learningRate, correctedMovingAverageSqr,
                correctedMovingAverageSqr, correctedMovingAverageSqr.length);
        VectorOperations.vectorToVectorScalarMultiplication(correctedMovingAverage, correctedMovingAverageSqr,
                correctedMovingAverage, correctedMovingAverage.length);
    }
}
