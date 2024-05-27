package com.tornadoml.cpu;

public class NoActivation implements ActivationFunction {
  @Override
  public void value(float[] input, float[] result, int length) {
    assert input.length >= length;
    assert result.length >= length;

    System.arraycopy(input, 0, result, 0, length);
  }

  @Override
  public void derivative(float[] input, float[] result, int length) {
    assert input.length >= length;
    assert result.length >= length;

    for (int i = 0; i < length; i++) {
      result[i] = 1.0f;
    }

  }

  @Override
  public void initWeighs(float[] weights, int inputSize) {
    LeakyLeRU.initWeights(weights, inputSize, -1);
  }
}
