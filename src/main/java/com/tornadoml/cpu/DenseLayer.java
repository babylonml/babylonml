package com.tornadoml.cpu;

public final class DenseLayer implements TrainableLayer {
    private static final ThreadLocal<float[]> outputXBatchSizeBuffer = new ThreadLocal<>();
    private static final ThreadLocal<float[]> firstInputXBatchSizeBuffer = new ThreadLocal<>();
    private static final ThreadLocal<float[]> secondInputXBatchSizeBuffer = new ThreadLocal<>();
    private static final ThreadLocal<float[]> weightsSizeBuffer = new ThreadLocal<>();

    private final float[] weights;
    private final float[] biases;

    private float[] bestWeights;
    private float[] bestBiases;

    private final ActivationFunction activationFunction;

    private final int inputSize;
    private final int outputSize;

    private final WeightsOptimizer optimizer;

    public DenseLayer(int inputSize, int outputSize, ActivationFunction activationFunction,
                      WeightsOptimizer.OptimizerType optimizerType) {
        this.activationFunction = activationFunction;
        weights = new float[inputSize * outputSize];
        biases = new float[outputSize];

        this.inputSize = inputSize;
        this.outputSize = outputSize;

        activationFunction.initWeighs(weights, inputSize);
        activationFunction.initWeighs(biases, inputSize);

        optimizer = WeightsOptimizer.instance(optimizerType, inputSize * outputSize, outputSize);
    }


    @Override
    public void predict(float[] input, float[] prediction, int batchSize) {
        assert input.length >= inputSize * batchSize;
        assert prediction.length >= outputSize * batchSize;

        //w * x
        MatrixOperations.matrixToMatrixMultiplication(weights, 0,
                outputSize, inputSize, input, 0, inputSize, batchSize, prediction);

        var biasesBuffer = getOutputXBatchSizeBuffer(batchSize * outputSize);
        //broadcast biases
        MatrixOperations.broadcastVectorToMatrix(biases, biasesBuffer, outputSize, batchSize);

        //w * x + b
        VectorOperations.addVectorToVector(prediction, biasesBuffer, prediction, batchSize * outputSize);
        //g(w * x + b)
        activationFunction.value(prediction, prediction, batchSize * outputSize);
    }

    @Override
    public void forwardTraining(float[] input, int inputOffset, float[] activationArgumentOutput, float[] predictionOutput,
                                int batchSize) {
        //w * x
        MatrixOperations.matrixToMatrixMultiplication(weights, 0,
                outputSize, inputSize, input, inputOffset, inputSize,
                batchSize, activationArgumentOutput);

        var buffer = getOutputXBatchSizeBuffer(batchSize * outputSize);
        //broadcast biases
        MatrixOperations.broadcastVectorToMatrix(biases, buffer, outputSize, batchSize);

        //w * x + b
        VectorOperations.addVectorToVector(activationArgumentOutput, buffer, activationArgumentOutput,
                batchSize * outputSize);

        //g(w * x + b)
        activationFunction.value(activationArgumentOutput, predictionOutput, batchSize * outputSize);
    }

    @Override
    public void backwardLastLayer(float[] input, float[] previousLayerActivationArgumentInput,
                                  float[] currentLayerActivationArgumentInput,
                                  float[] currentLayerErrorsInput, float[] previousLayerErrorsOutput,
                                  float[] calculatedWeightsGradientOutput,
                                  float[] calculatedBiasesGradientOutput,
                                  int batchSize) {
        //y - output
        //dl/dz  - errors
        //[n] - current layer, [n-1] - previous layer
        //w - weights
        //b - biases
        //z = w * a + b
        //a[n-1] - input
        //a[n] - output

        var outputBatchSizeBuffer = getOutputXBatchSizeBuffer(batchSize * outputSize);
        //g'(z[n])
        activationFunction.derivative(currentLayerActivationArgumentInput, outputBatchSizeBuffer, batchSize * outputSize);
        //dl/dz[n] = dL/dy * g'(z[n])
        VectorOperations.vectorToVectorScalarMultiplication(currentLayerErrorsInput, outputBatchSizeBuffer,
                currentLayerErrorsInput, outputSize * batchSize);

        //dL/dw[n] = dL/dz[n] * a[n-1]^T
        calculateWeightsDelta(input, 0, currentLayerErrorsInput, calculatedWeightsGradientOutput, calculatedBiasesGradientOutput,
                batchSize);

        //dL/dz[n-1] = w[n]^T * dL/dz[n] * g'(z[n-1])
        calculatePreviousLayerError(currentLayerErrorsInput, previousLayerActivationArgumentInput, previousLayerErrorsOutput,
                batchSize);
    }

    @Override
    public void backwardSingleLayerNoError(float[] input,
                                           float[] currentLayerActivationArgumentInput,
                                           float[] currentLayerErrorInput, float[] calculatedWeightsGradientOutput,
                                           float[] calculatedBiasesGradientOutput,
                                           int batchSize) {
        //y - output
        //dl/dz  - errors
        //[n] - current layer, [n-1] - previous layer
        //w - weights
        //b - biases
        //z = w * a + b
        //a[n-1] - input
        //a[n] - output


        var outputBatchSizeBuffer = getOutputXBatchSizeBuffer(batchSize * outputSize);
        //g'(z[n])
        activationFunction.derivative(currentLayerActivationArgumentInput, outputBatchSizeBuffer, batchSize * outputSize);
        //dl/dz[n] = dL/dy * g'(z[n])
        VectorOperations.vectorToVectorScalarMultiplication(currentLayerErrorInput, outputBatchSizeBuffer,
                currentLayerErrorInput, outputSize * batchSize);

        //dL/dw[n] = dL/dz[n] * a[n-1]^T
        calculateWeightsDelta(input, 0, currentLayerErrorInput, calculatedWeightsGradientOutput, calculatedBiasesGradientOutput,
                batchSize);
    }


    @Override
    public void backwardMiddleLayer(float[] input,
                                    float[] currentLayerErrorsInput,
                                    float[] previousLayerActivationArgumentInput,
                                    float[] previousLayerErrorsOutput, float[] weightsGradientOutput, float[] biasesGradientOutput,
                                    int batchSize) {
        //y - output
        //dl/dz  - errors
        //[n] - current layer, [n-1] - previous layer
        //w - weights
        //b - biases
        //z = w * a + b
        //a[n-1] - input
        //a[n] - output

        //dL/dw[n] = dL/dz[n] * a[n-1]^T
        calculateWeightsDelta(input, 0, currentLayerErrorsInput, weightsGradientOutput, biasesGradientOutput, batchSize);

        //dL/dz[n-1] = w[n]^T * dL/dz[n] * g'(z[n-1])
        calculatePreviousLayerError(currentLayerErrorsInput, previousLayerActivationArgumentInput, previousLayerErrorsOutput,
                batchSize);
    }

    private void calculatePreviousLayerError(float[] currentLayerErrors,
                                             float[] previousLayerActivationArgument,
                                             float[] previousLayerErrors,
                                             int batchSize) {
        var weightsSizeBuffer = getWeightsSizeBuffer(inputSize * outputSize);
        //w[n]^T
        MatrixOperations.transposeMatrix(weights, 0, outputSize, inputSize, weightsSizeBuffer);

        var firstInputBatchSizeBuffer = getFirstInputXBatchSizeBuffer(batchSize * inputSize);
        //w[n]^T * dL/dz[n]
        MatrixOperations.matrixToMatrixMultiplication(weightsSizeBuffer, 0, inputSize, outputSize,
                currentLayerErrors, 0, outputSize, batchSize, firstInputBatchSizeBuffer);

        var secondInputBatchSizeBuffer = getSecondInputXBatchSizeBuffer(batchSize * inputSize);
        //g'(z[n-1])
        activationFunction.derivative(previousLayerActivationArgument, secondInputBatchSizeBuffer,
                batchSize * inputSize);

        //w[n]^T * dL/dz[n] * g'(z[n-1])
        VectorOperations.vectorToVectorScalarMultiplication(firstInputBatchSizeBuffer, secondInputBatchSizeBuffer,
                previousLayerErrors, inputSize * batchSize);
    }

    private void calculateWeightsDelta(float[] input, int inputOffset,
                                       float[] errors,
                                       float[] weightsDelta,
                                       float[] biasesDelta,
                                       int batchSize) {
        var inputBatchSizeBuffer = getFirstInputXBatchSizeBuffer(batchSize * inputSize);
        //a[n-1]^T
        MatrixOperations.transposeMatrix(input, inputOffset, inputSize, batchSize, inputBatchSizeBuffer);
        //dL/dw[n] = dL/dz[n] * a[n-1]^T
        MatrixOperations.matrixToMatrixMultiplication(errors, 0, outputSize, batchSize,
                inputBatchSizeBuffer, 0, batchSize, inputSize, weightsDelta);

        System.arraycopy(errors, 0, biasesDelta, 0, outputSize * batchSize);
    }

    @Override
    public void backwardFirstLayer(float[] input, int inputOffset, float[] currentLayerErrorsInput, float[] weightsGradientOutput,
                                   float[] biasesGradientOutput,
                                   int batchSize) {
        //y - output
        //dl/dz  - errors
        //[n] - current layer, [n-1] - previous layer
        //w - weights
        //b - biases
        //z = w * a + b
        //a[n-1] - input
        //a[n] - output

        //dL/dw[n] = dL/dz[n] * a[n-1]^T
        calculateWeightsDelta(input, inputOffset, currentLayerErrorsInput, weightsGradientOutput, biasesGradientOutput, batchSize);
    }

    @Override
    public int getInputSize() {
        return inputSize;
    }

    @Override
    public int getOutputSize() {
        return outputSize;
    }

    @Override
    public void updateWeightsAndBiases(float[] weightsGradient, float[] biasesGradient, float learningRate) {
        optimizer.optimize(weights, weightsGradient, inputSize * outputSize, biases, biasesGradient,
                outputSize, learningRate);
    }

    @Override
    public void saveWeightsAndBiases() {
        bestWeights = weights.clone();
        bestBiases = biases.clone();
    }

    @Override
    public void restoreBestWeightsAndBiases() {
        System.arraycopy(bestWeights, 0, weights, 0, weights.length);
        System.arraycopy(bestBiases, 0, biases, 0, biases.length);
    }

    public float[] getWeights() {
        return weights;
    }

    public float[] getBiases() {
        return biases;
    }

    private static float[] getOutputXBatchSizeBuffer(int size) {
        var buffer = outputXBatchSizeBuffer.get();

        if (buffer == null || buffer.length < size) {
            buffer = new float[size];
            outputXBatchSizeBuffer.set(buffer);
        }

        return buffer;
    }

    private static float[] getWeightsSizeBuffer(int size) {
        var buffer = weightsSizeBuffer.get();

        if (buffer == null || buffer.length < size) {
            buffer = new float[size];
            weightsSizeBuffer.set(buffer);
        }

        return buffer;
    }

    private static float[] getFirstInputXBatchSizeBuffer(int size) {
        var buffer = firstInputXBatchSizeBuffer.get();

        if (buffer == null || buffer.length < size) {
            buffer = new float[size];
            firstInputXBatchSizeBuffer.set(buffer);
        }

        return buffer;
    }

    private static float[] getSecondInputXBatchSizeBuffer(int size) {
        var buffer = secondInputXBatchSizeBuffer.get();

        if (buffer == null || buffer.length < size) {
            buffer = new float[size];
            secondInputXBatchSizeBuffer.set(buffer);
        }

        return buffer;
    }

    @Override
    public int getWeightsSize() {
        return inputSize * outputSize;
    }
}
