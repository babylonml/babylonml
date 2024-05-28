package com.tornadoml.cpu;

public interface WeightsOptimizer {
    void optimize(float[] weights, float[] weightsGradient, int weightsLength, float[] biases,
                  float[] biasesGradient, int biasesLength, float learningRate);

    static WeightsOptimizer instance(OptimizerType optimizerType, int weightsSize, int biasesSize) {
        return switch (optimizerType) {
            case SIMPLE -> new SimpleGradientDescentOptimizer();
            case ADAM -> new AdamOptimizer(weightsSize, biasesSize);
            case AMS_GRAD -> new AMSGradOptimizer(weightsSize, biasesSize);
        };
    }

    enum OptimizerType {
        SIMPLE,
        ADAM,
        AMS_GRAD
    }
}
