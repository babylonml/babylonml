package com.tornadoml.cpu;

public class CategoricalCrossEntropyFunction implements CostFunction {
    @Override
    public float value(float[] output, int outputOffset, float[] target, int targetOffset, int length, int batchSize) {
        var sum = 0.0f;
        for (int i = 0; i < length; i++) {
            sum += -target[targetOffset + i] * (float) Math.log(output[outputOffset + i]);
        }
        return sum / batchSize;
    }

    @Override
    public void derivative(float[] output, int outputOffset, float[] target, int targetOffset,
                           float[] result, int resultOffset, int length) {
    }
}
