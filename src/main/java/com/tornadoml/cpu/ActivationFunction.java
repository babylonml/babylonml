package com.tornadoml.cpu;

/**
 * Activation function interface.
 */
public interface ActivationFunction {
    /**
     * Applies the activation function to the input array and stores the result in the result array.
     * Passed in array could have a vector or matrix in flatten form, when it is a matrix it is in row major order.
     *
     * @param input  input array could be a vector or matrix in flatten form, when it is a matrix it is in row major order
     * @param result result array could be a vector or matrix in flatten form, when it is a matrix it is in row major order
     * @param length length of the both arrays.
     */
    void value(float[] input, float[] result, int length);

    /**
     * Applies the derivative of the activation function to the input array and stores the result in the result array.
     * Passed in array could have a vector or matrix in flatten form, when it is a matrix it is in row major order.
     *
     * @param input  input array could be a vector or matrix in flatten form, when it is a matrix it is in row major order
     * @param result result array could be a vector or matrix in flatten form, when it is a matrix it is in row major order
     * @param length length of the both arrays.
     */
    void derivative(float[] input, float[] result, int length);

    /**
     * Initializes the weights of the layer that uses given activation function.
     * This method is used to initialize both weights and biases.
     *
     * @param weights   weights to be initialized
     * @param inputSize size of the input
     */
    void initWeighs(float[] weights, int inputSize);
}
