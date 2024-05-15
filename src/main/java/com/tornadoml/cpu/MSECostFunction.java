package com.tornadoml.cpu;

public final class MSECostFunction implements CostFunction {
    @Override
    public float value(float[] output, int inputOffset, float[] target, int targetOffset, int length, int batchSize) {
        float sum = 0;
        for (int i = 0; i < length; i++) {
            var value = output[i + inputOffset] - target[i + targetOffset];
            sum += value * value;
        }
        return sum / batchSize;
    }

    @Override
    public void derivative(float[] output, int inputOffset, float[] target, int targetOffset, float[] result,
                           int resultOffset, int length) {
        for (int i = inputOffset; i < length; i++) {
            result[i + resultOffset] = 2.0f * (output[i] - target[i + targetOffset]);
        }
    }
}
