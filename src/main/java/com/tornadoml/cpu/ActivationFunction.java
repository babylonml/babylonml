package com.tornadoml.cpu;

public interface ActivationFunction {
    void value(float[] input, float[] result);
    void derivative(float[] input, float[] result);
    void initWeighs(float[] weights, int inputSize);
}
