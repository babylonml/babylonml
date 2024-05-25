package com.tornadoml.cpu;

public final class SoftMaxLayer implements NonTrainableLayer {

    private final int inputSize;

    public SoftMaxLayer(int inputSize) {
        this.inputSize = inputSize;
    }

    @Override
    public void predict(float[] input, float[] prediction, int batchSize) {
        MatrixOperations.softMaxByColumns(input, inputSize, batchSize, prediction);
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
