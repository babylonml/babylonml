package com.tornadoml.cpu;

public interface NonTrainableLayer extends Layer {
    void backwardLastLayer(float[] input,
        float[] target, float[] costFunctionDerivative, int miniBatchSize);
}
