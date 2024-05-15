package com.tornadoml.cpu;

public interface CostFunction {
  float value(float[] output, int outputOffset, float[] target, int targetOffset , int length, int batchSize);
  void derivative(float[] output, int outputOffset, float[] target, int targetOffset, float [] result, int resultOffset, int length);
}
