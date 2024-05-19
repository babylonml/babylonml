package com.tornadoml.cpu;

public interface ActivationFunction {
    void value(float[] input, float[] result, int length);
    void derivative(float[] input, float[] result, int length);
    void initWeighs(float[] weights, int inputSize);
}
