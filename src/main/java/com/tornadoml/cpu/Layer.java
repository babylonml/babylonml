package com.tornadoml.cpu;

public interface Layer {
    void predict(float[] input, float[] prediction, int batchSize);

    int getInputSize();

    int getOutputSize();
}
