package com.tornadoml.cpu;

/**
 * Interface for layers that do not have weights and biases and so can not be trained.
 * As of now the layer can be only the output layer of the network.
 */
public interface NonTrainableLayer extends Layer {
    /**
     * Backward pass for the last layer of the network.
     * This method is used to calculate the error output of the previous layer during the backpropagation.
     * <p>
     * Size of passed in arrays may be bigger than needed, so never rely on the size of the arrays and use such
     * parameters as batch size, input size and output size, array offsets to calculate the size of the arrays.
     *
     * @param input                    Layer input. It is a matrix in flatten form, with a row major order.
     *                                 Amount of rows equals to input size and amount of columns equals to batch size.
     * @param costFunctionInput        Input of the cost function. It is a matrix in flatten form, with a row major order.
     *                                 Amount of rows equals to output size and amount of columns equals to batch size.
     * @param previousLayerErrorOutput Error output of the previous layer. It is a matrix in flatten form, with a row major order.
     *                                 Amount of rows equals to input size and amount of columns equals to batch size.
     * @param batchSize                Size of the batch.
     */
    void backwardLastLayer(float[] input,
                           float[] costFunctionInput, float[] previousLayerErrorOutput, int batchSize);
}
