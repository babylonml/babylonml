package com.tornadoml.cpu;

public class Tanh implements ActivationFunction {
  @Override
  public void value(float[] input, float[] result, int length) {
    assert input.length >= length;
    assert result.length >= length;

    for (int i = 0; i < length; i++) {
      result[i] = (float) Math.tanh(input[i]);
    }
  }

  @Override
  public void derivative(float[] input, float[] result, int length) {
    assert input.length >= length;
    assert result.length >= length;

    for (int i = 0; i < length; i++) {
      result[i] = 1 - input[i] * input[i];
    }
  }

  @Override
  public void initWeighs(float[] weights, int inputSize) {
    LeakyLeRU.initWeights(weights, inputSize, -1);
  }
}
