package com.tornadoml.cpu;

public class AMSGradOptimizer implements WeightsOptimizer {
    private static final ThreadLocal<float[]> calculationBuffer = new ThreadLocal<>();

    private final float[] avgWeightsMovement;
    private final float[] avgBiasesMovement;

    private final float[] avgWeightsMovementSqr;
    private final float[] avgBiasesMovementSqr;

    private final float[] correctedAvgWeightsMovementSqr;
    private final float[] correctedAvgBiasesMovementSqr;


    public AMSGradOptimizer(int weightsSize, int biasesSize) {
        avgWeightsMovement = new float[weightsSize];
        avgBiasesMovement = new float[biasesSize];
        avgWeightsMovementSqr = new float[weightsSize];
        avgBiasesMovementSqr = new float[biasesSize];
        correctedAvgWeightsMovementSqr = new float[weightsSize];
        correctedAvgBiasesMovementSqr = new float[biasesSize];
    }

    @Override
    public void optimize(float[] weights, float[] weightsGradient, int weightsLength, float[] biases,
                         float[] biasesGradient, int biasesLength, float learningRate) {
        var calculationBuffer = AMSGradOptimizer.calculationBuffer.get();
        if (calculationBuffer == null || calculationBuffer.length < weightsLength) {
            calculationBuffer = new float[weightsLength];
            AMSGradOptimizer.calculationBuffer.set(calculationBuffer);
        }

        AdamOptimizer.updateAvgMovement(weightsGradient, avgWeightsMovement, avgWeightsMovementSqr, weightsLength,
                calculationBuffer);
        correctAvgMovementSqr(avgWeightsMovementSqr, correctedAvgWeightsMovementSqr);
        calculateCorrections(avgWeightsMovement, correctedAvgWeightsMovementSqr, calculationBuffer, learningRate);
        VectorOperations.addVectorToVector(weights, calculationBuffer, weights, weightsLength);

        AdamOptimizer.updateAvgMovement(biasesGradient, avgBiasesMovement, avgBiasesMovementSqr, biasesLength,
                calculationBuffer);
        correctAvgMovementSqr(avgBiasesMovementSqr, correctedAvgBiasesMovementSqr);
        calculateCorrections(avgBiasesMovement, correctedAvgBiasesMovementSqr, calculationBuffer, learningRate);
        VectorOperations.addVectorToVector(biases, calculationBuffer, biases, biasesLength);
    }

    private static void correctAvgMovementSqr(float[] avgWeightsMovementSqr, float[] correctedAvgWeightsMovementSqr) {
        VectorOperations.maxBetweenVectorElements(avgWeightsMovementSqr, correctedAvgWeightsMovementSqr,
                correctedAvgWeightsMovementSqr, correctedAvgWeightsMovementSqr.length);
    }

    private static void calculateCorrections(float[] movingAverage, float[] correctedMovingAverageSqr,
                                             float[] calculationBuffer, float learningRate) {
        VectorOperations.vectorElementsSqrt(correctedMovingAverageSqr, calculationBuffer, correctedMovingAverageSqr.length);
        VectorOperations.addScalarToVector(1e-8f, calculationBuffer, calculationBuffer, correctedMovingAverageSqr.length);
        VectorOperations.divideScalarOnVectorElements(-learningRate, calculationBuffer,
                calculationBuffer, correctedMovingAverageSqr.length);
        VectorOperations.vectorToVectorScalarMultiplication(movingAverage, calculationBuffer,
                calculationBuffer, movingAverage.length);
    }
}
