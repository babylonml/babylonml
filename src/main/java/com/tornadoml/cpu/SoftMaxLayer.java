package com.tornadoml.cpu;

public final class SoftMaxLayer implements NonTrainableLayer {

  private final int inputSize;

  public SoftMaxLayer(int inputSize) {
    this.inputSize = inputSize;
  }

  @Override
  public void predict(float[] input, float[] prediction, int batchSize) {
    VectorOperations.vectorElementsExp(input, prediction, inputSize * batchSize);

    for (int i = 0, offset = 0; i < batchSize; i++, offset += inputSize) {
      var sum = VectorOperations.sumVectorElements(prediction, offset, inputSize);
      VectorOperations.multiplyVectorToScalar(prediction, offset, 1.0f / sum, prediction, offset,
              inputSize);
    }
  }

  @Override
  public int getInputSize() {
    return inputSize;
  }

  @Override
  public int getOutputSize() {
    return inputSize;
  }

  @Override
  public void backwardLastLayer(float[] input, float[] target, float[] costFunctionDerivative,
      int miniBatchSize) {
    VectorOperations.subtractVectorFromVector(input, target, costFunctionDerivative,
        inputSize * miniBatchSize);

  }
}
