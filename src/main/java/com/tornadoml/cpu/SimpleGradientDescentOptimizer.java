package com.tornadoml.cpu;

public class SimpleGradientDescentOptimizer implements WeightsOptimizer {
    private static final ThreadLocal<float[]> calculationBuffer = new ThreadLocal<>();

    @Override
    public void optimize(float[] weights, float[] weightsGradient, int weightsLength,
                         float[] biases, float[] biasesGradient, int biasesLength, float learningRate) {
        var calculationBuffer = getCalculationBuffer(weightsLength);

        VectorOperations.multiplyVectorToScalar(weightsGradient, 0, -learningRate,
                calculationBuffer, 0, weightsLength);
        VectorOperations.addVectorToVector(weights, 0, calculationBuffer, 0, weights,
                0, weightsLength);

        VectorOperations.multiplyVectorToScalar(biasesGradient, 0, -learningRate,
                calculationBuffer, 0, biasesLength);
        VectorOperations.addVectorToVector(biases, 0, calculationBuffer, 0, biases,
                0, biasesLength);
    }

    private static float[] getCalculationBuffer(int weightsLength) {
        var calculationBuffer = SimpleGradientDescentOptimizer.calculationBuffer.get();

        if (calculationBuffer == null || calculationBuffer.length < weightsLength) {
            calculationBuffer = new float[weightsLength];
            SimpleGradientDescentOptimizer.calculationBuffer.set(calculationBuffer);
        }

        return calculationBuffer;
    }
}
