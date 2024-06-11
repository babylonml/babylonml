package com.tornadoml.cpu;

public final class SoftMaxLayer implements NonTrainableLayer {

    private final int inputSize;

    public SoftMaxLayer(int inputSize) {
        this.inputSize = inputSize;
    }

    @Override
    public void predict(float[] input, float[] prediction, int batchSize) {
        MatrixOperations.softMaxByColumns(input, 0, inputSize, batchSize, prediction, 0);
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
    public void backwardLastLayer(float[] input, float[] costFunctionInput, float[] previousLayerErrorOutput,
                                  int batchSize) {
        VectorOperations.subtractVectorFromVector(input, 0, costFunctionInput, 0, previousLayerErrorOutput,
                0, inputSize * batchSize);

    }
}
