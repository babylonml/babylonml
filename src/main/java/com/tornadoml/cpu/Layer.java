package com.tornadoml.cpu;

/**
 * Base interface for all layers in neural network.
 */
public interface Layer {
    /**
     * Predicts the output of the layer given the input.
     * Size of the input array may be bigger than needed, so never rely on the size of the input array and use such
     * parameters as batch size, input size and output size to calculate the size of the input and output arrays.
     *
     * @param input      input array is a matrix in flatten form, with a row major order.
     *                   Amount of rows equals to input size and amount of columns equals to batch size.
     * @param prediction prediction array is a matrix in flatten form, it is in row major order.
     *                   Amount of rows equals to output size and amount of columns equals to batch size.
     * @param batchSize  size of the batch.
     */
    void predict(float[] input, float[] prediction, int batchSize);

    /**
     * Returns the size of the input.
     *
     * @return size of the input.
     */
    int getInputSize();

    /**
     * Returns the size of the output.
     *
     * @return size of the output.
     */
    int getOutputSize();
}
