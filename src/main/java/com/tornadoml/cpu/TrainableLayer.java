package com.tornadoml.cpu;

public interface TrainableLayer extends Layer {
    void updateWeightsAndBiases(float[] weightsDelta, float[] biasesDelta, float learningRate);

    void saveBestWeightsAndBiases();

    void restoreBestWeightsAndBiases();

    void forwardTraining(float[] input, int inputOffset, float[] activationArgument, float[] prediction,
                          int miniBatchSize);

    void backwardLastLayer(float[] input, float[] previousLayerActivationArgument,
                           float[] currentLayerActivationArgument,
                           float[] currentLayerErrors, float[] previousLayerErrors, float[] calculatedWeightsDelta,
                           float[] calculatedBiasesDelta,
                           int miniBatchSize);

    void backwardLastLayerNoError(float[] input,
                                  float[] currentLayerActivationArgument,
                                  float[] costFunctionDerivative, float[] calculatedWeightsDelta,
                                  float[] calculatedBiasesDelta,
                                  int miniBatchSize);

    void backwardMiddleLayer(float[] input,
                             float[] errors,
                             float[] previousLayerActivationArgument,
                             float[] previousLayerErrors, float[] weightsDelta, float[] biasesDelta,
                             int miniBatchSize);

    void backwardZeroLayer(float[] input, int inputOffset, float[] errors, float[] weightsDelta,
                           float[] biasesDelta,
                           int miniBatchSize);
}
