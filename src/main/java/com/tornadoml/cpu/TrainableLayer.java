package com.tornadoml.cpu;

public interface TrainableLayer extends Layer {
    /**
     * Called at the end of the back propagation to update the weights and biases of the layer.
     * All gradient values are averaged over the bath size, you don't need to take batch size into account.
     * <p>
     * By design each layer should not update its weights and biases itself but rely on {@link  WeightsOptimizer}.
     * The simples form of the optimizer {@link SimpleGradientDescentOptimizer} multiplies the gradient by the learning rate
     * and subtracts it from the weights and biases.
     *
     * @param weightsGradient values of the gradient of the cost function with respect to the weights of the layer.
     * @param biasesGradient  values of the gradient of the cost function with respect to the biases of the layer.
     * @param learningRate    learning rate
     */
    void updateWeightsAndBiases(float[] weightsGradient, float[] biasesGradient, float learningRate);

    /**
     * Saves the current weights and biases of the layer to the best weights and biases, so they could be restored later by
     * calling {@link #restoreBestWeightsAndBiases()}.
     * <p>
     * This method is used during the training to save the best weights and biases of the layer.
     */
    void saveWeightsAndBiases();

    /**
     * Restores the best weights and biases of the layer that were saved by calling {@link #saveWeightsAndBiases()}.
     * <p>
     * This method is used during the training to restore the best weights and biases of the layer.
     */
    void restoreBestWeightsAndBiases();

    /**
     * Performance optimized form of the forward propagation step for the training phase.
     * <p>
     * Size of passed in arrays may be bigger than needed, so never rely on the size of the arrays and use such
     * parameters as batch size, input size and output size, array offsets to calculate the size of the arrays.
     *
     * @param input                    Layer input.
     *                                 Matrix in flatten form, with a row major order.
     *                                 Amount of rows equals to input size and amount of columns
     *                                 equals to batch size.
     * @param inputOffset              offset in the input array from where the input starts
     * @param activationArgumentOutput Activation argument array . That is
     *                                 an output parameter, value that is passed to the activation function.
     *                                 This value is used in the back propagation to calculate the derivative
     *                                 of the activation function and error for the given layer during the back propagation.
     *                                 This is a matrix in flatten form, it is in row major order.
     *                                 Amount of rows equals to output size and amount of columns equals to batch size.
     * @param predictionOutput         Prediction array is a matrix in flatten form, it is in row major order.
     *                                 It is an output parameter. Result of the forward propagation.
     *                                 Amount of rows equals to output size and amount of columns equals to batch size.
     * @param batchSize                Size of the batch.
     */
    void forwardTraining(float[] input, int inputOffset,
                         float[] activationArgumentOutput, float[] predictionOutput,
                         int batchSize);

    /**
     * Performance optimized form of the backward propagation step for the training phase.
     * This method is called only for the last layer in neural network.
     * <p>
     * Size of passed in arrays may be bigger than needed, so never rely on the size of the arrays and use such
     * parameters as batch size, input size and output size, to calculate the size of the arrays.
     *
     * @param input                                Layer input.
     *                                             Matrix in flatten form, with a row major order.
     *                                             Amount of rows equals to input size and amount of columns
     *                                             equals to batch size.
     * @param previousLayerActivationArgumentInput Activation argument array of the previous layer.
     *                                             It is a matrix in flatten form, it is in row major order.
     *                                             It is input parameter calculated during forward propagation.
     *                                             Amount of rows equals to input size and amount of columns equals to batch size.
     * @param currentLayerActivationArgumentInput  Activation argument array of the current layer. It is a matrix in flatten form,
     *                                             it is in row major order. Amount of rows equals to output size and amount of columns
     *                                             equals to batch size. It is input parameter calculated during forward propagation.
     * @param currentLayerErrorsInput              Errors of the current layer calculated by next
     *                                             layer during backward propagation.
     *                                             It is a matrix in flatten form, it is in row major order.
     *                                             It is an input parameter.
     *                                             Amount of rows equals to output size and amount of columns
     *                                             equals to batch size.
     * @param previousLayerErrorsOutput            Errors of the previous layer. It is an output parameter. It is a matrix in flatten form,
     *                                             it is in row major order. Amount of rows equals to input size and amount of columns
     *                                             equals to batch size.
     * @param calculatedWeightsGradientOutput      Calculated weights gradient. It is an output parameter. It is a matrix in flatten form,
     *                                             it is in row major order. Amount of rows equals to output size and amount of columns
     *                                             equals to input size.
     * @param calculatedBiasesGradientOutput       Calculated biases gradient. It is an output parameter.
     *                                             It is a matrix where every gradient is broadcast amount of times equals to the batch size.
     *                                             It is in flatten form, it is in row major order.
     *                                             Amount of rows equals to output size and amount of columns
     *                                             equals to batch size.
     * @param batchSize                            Size of the batch.
     */
    void backwardLastLayer(float[] input, float[] previousLayerActivationArgumentInput,
                           float[] currentLayerActivationArgumentInput,
                           float[] currentLayerErrorsInput, float[] previousLayerErrorsOutput,
                           float[] calculatedWeightsGradientOutput,
                           float[] calculatedBiasesGradientOutput,
                           int batchSize);

    /**
     * Performance optimized form of the backward propagation step for the training phase.
     * <p>
     * This method is used when the layer the only layer in the neural network.
     *
     * <p>
     * Size of passed in arrays may be bigger than needed, so never rely on the size of the arrays and use such
     * parameters as batch size, input size and output size, to calculate the size of the arrays.
     *
     * @param input                               Layer input.
     *                                            Matrix in flatten form, with a row major order. Amount of rows equals to input size and amount of columns
     *                                            equals to batch size.
     * @param currentLayerActivationArgumentInput Activation argument array of the current layer. It is a matrix in flatten form,
     *                                            it is in row major order. Amount of rows equals to output size and amount of columns
     *                                            equals to batch size.
     *                                            It is input parameter calculated during forward propagation.
     * @param currentLayerErrorInput              Errors of the current layer calculated by next layer during backward propagation.
     *                                            It is a matrix in flatten form, it is in row major order.
     *                                            It is an input parameter. Amount of rows equals to output size and amount of columns
     *                                            equals to batch size.
     * @param calculatedWeightsGradientOutput     Calculated weights gradient. It is an output parameter. It is a matrix in flatten form,
     *                                            it is in row major order. Amount of rows equals to output size and amount of columns
     *                                            equals to input size.
     * @param calculatedBiasesGradientOutput      Calculated biases gradient. It is an output parameter. It is a matrix where every gradient is broadcast
     *                                            amount of times equals to the batch size. It is in flatten form, it is in row major order.
     *                                            Amount of rows equals to output size and amount of columns equals to batch size.
     * @param batchSize                           Size of the batch.
     */
    void backwardSingleLayerNoError(float[] input,
                                    float[] currentLayerActivationArgumentInput,
                                    float[] currentLayerErrorInput, float[] calculatedWeightsGradientOutput,
                                    float[] calculatedBiasesGradientOutput,
                                    int batchSize);

    /**
     * Performance optimized form of the backward propagation step for the training phase.
     * <p>
     * Size of passed in arrays may be bigger than needed, so never rely on the size of the arrays and use such
     * parameters as batch size, input size and output size, to calculate the size of the arrays.
     *
     * @param input                                Layer input.
     *                                             Matrix in flatten form, with a row major order.
     *                                             Amount of rows equals to input size
     *                                             and amount of columns equals to batch size.
     * @param currentLayerErrorsInput              Errors of the current layer calculated by next layer during backward propagation.
     *                                             It is a matrix in flatten form, it is in row major order.
     *                                             It is an input parameter. Amount of rows equals to output size and amount of columns
     *                                             equals to batch size.
     * @param previousLayerActivationArgumentInput Activation argument array of the previous layer. It is a matrix in flatten form,
     *                                             it is in row major order. Amount of rows equals to input size and amount of columns
     *                                             equals to batch size. It is input parameter calculated during forward propagation.
     * @param previousLayerErrorsOutput            Errors of the previous layer. It is an output parameter. It is a matrix in flatten form,
     *                                             it is in row major order. Amount of rows equals to input size and amount of columns
     *                                             equals to batch size.
     * @param weightsGradientOutput                Calculated weights gradient. It is an output parameter. It is a matrix in flatten form,
     *                                             it is in row major order. Amount of rows equals to output size and amount of columns
     *                                             equals to input size.
     * @param biasesGradientOutput                 Calculated biases gradient. It is an output parameter.
     *                                             It is a matrix where every gradient is broadcast
     *                                             amount of times equals to the batch size.
     *                                             It is in flatten form, it is in row major order.
     *                                             Amount of rows equals to output size and amount of columns
     *                                             equals to batch size.
     * @param batchSize                            Size of the batch.
     */
    void backwardMiddleLayer(float[] input,
                             float[] currentLayerErrorsInput,
                             float[] previousLayerActivationArgumentInput,
                             float[] previousLayerErrorsOutput, float[] weightsGradientOutput, float[] biasesGradientOutput,
                             int batchSize);

    /**
     * Performance optimized form of the backward propagation step for the training phase.
     * <p>
     * This method is used when the layer is the last layer in the neural network. So no error calculation is needed.
     * <p>
     * Size of passed in arrays may be bigger than needed, so never rely on the size of the arrays and use such
     * parameters as batch size, input size and output size, to calculate the size of the arrays.
     *
     * @param input                   Layer input. Matrix in flatten form, with a row major order.
     *                                Amount of rows equals to input size and amount of columns equals to batch size.
     * @param inputOffset             offset in the input array from where the input starts.
     * @param currentLayerErrorsInput Errors of the current layer calculated by next layer during backward propagation.
     *                                It is a matrix in flatten form, it is in row major order.
     *                                It is an input parameter. Amount of rows equals to output size and amount of columns
     *                                equals to batch size.
     * @param weightsGradientOutput   Calculated weights gradient. It is an output parameter. It is a matrix in flatten form,
     *                                it is in row major order. Amount of rows equals to output size and amount of columns
     *                                equals to input size.
     * @param biasesGradientOutput    Calculated biases gradient. It is an output parameter.
     *                                It is a matrix where every gradient is broadcast
     *                                amount of times equals to the batch size. It is in flatten form, it is in row major order.
     *                                Amount of rows equals to output size and amount of columns equals to batch size.
     * @param batchSize               Size of the batch.
     */
    void backwardLastLayer(float[] input, int inputOffset, float[] currentLayerErrorsInput, float[] weightsGradientOutput,
                           float[] biasesGradientOutput,
                           int batchSize);
}
