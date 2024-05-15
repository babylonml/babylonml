package com.tornadoml.cpu;

public final class DenseLayer implements TrainableLayer {
    private static final ThreadLocal<float[]> firstOutputXBatchSizeBuffer = new ThreadLocal<>();
    private static final ThreadLocal<float[]> secondOutputXBatchSizeBuffer = new ThreadLocal<>();
    private static final ThreadLocal<float[]> inputXBatchSizeBuffer = new ThreadLocal<>();
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
        assert input.length == inputSize;
        assert prediction.length == outputSize;

        //w * x
        MatrixOperations.matrixToMatrixMultiplication(weights, 0,
                outputSize, inputSize, input, 0, inputSize, 1, prediction);

        //w * x + b
        VectorOperations.addVectorToVector(prediction, biases, prediction, outputSize);

        //g(w * x + b)
        activationFunction.value(prediction, prediction);
    }

    @Override
    public void forwardTraining(float[] input, int inputOffset, float[] activationArgument, float[] prediction,
                                int miniBatchSize) {
        //w * x
        MatrixOperations.matrixToMatrixMultiplication(weights, 0,
                outputSize, inputSize, input, inputOffset, inputSize,
                miniBatchSize, activationArgument);

        var buffer = getFirstOutputXBatchSizeBuffer(miniBatchSize * outputSize);
        //broadcast biases
        MatrixOperations.broadcastVectorToMatrix(biases, buffer, outputSize, miniBatchSize);

        //w * x + b
        VectorOperations.addVectorToVector(activationArgument, buffer, activationArgument,
                miniBatchSize * outputSize);

        //g(w * x + b)
        activationFunction.value(activationArgument, prediction);
    }

    @Override
    public void backwardLastLayer(float[] input, float[] previousLayerActivationArgument,
                                  float[] currentLayerActivationArgument,
                                  float[] costFunctionDerivative, float[] calculatedWeightsDelta,
                                  float[] calculatedBiasesDelta,
                                  int miniBatchSize) {
        //y - output
        //dl/dz  - errors
        //[n] - current layer, [n-1] - previous layer
        //w - weights
        //b - biases
        //z = w * a + b
        //a[n-1] - input
        //a[n] - output

        var inputBatchSizeBuffer = getInputXBatchSizeBuffer(miniBatchSize * inputSize);
        //g'(z[n])
        activationFunction.derivative(currentLayerActivationArgument, inputBatchSizeBuffer);
        //dl/dz[n] = dL/dy * g'(z[n])
        VectorOperations.vectorToVectorScalarMultiplication(costFunctionDerivative, inputBatchSizeBuffer,
                costFunctionDerivative, inputSize * miniBatchSize);

        //dL/dw[n] = dL/dz[n] * a[n-1]^T
        calculateWeightsDelta(input, 0, costFunctionDerivative, calculatedWeightsDelta, calculatedBiasesDelta,
                miniBatchSize);

        //dL/dz[n-1] = w[n]^T * dL/dz[n] * g'(z[n-1])
        calculatePreviousLayerError(costFunctionDerivative, previousLayerActivationArgument,
                miniBatchSize);
    }

    @Override
    public void backwardLastLayerNoError(float[] input,
                                         float[] currentLayerActivationArgument,
                                         float[] costFunctionDerivative, float[] calculatedWeightsDelta,
                                         float[] calculatedBiasesDelta,
                                         int miniBatchSize) {
        //y - output
        //dl/dz  - errors
        //[n] - current layer, [n-1] - previous layer
        //w - weights
        //b - biases
        //z = w * a + b
        //a[n-1] - input
        //a[n] - output


        var outputBatchSizeBuffer = getSecondOutputXBatchSizeBuffer(miniBatchSize * outputSize);
        //g'(z[n])
        activationFunction.derivative(currentLayerActivationArgument, outputBatchSizeBuffer);
        //dl/dz[n] = dL/dy * g'(z[n])
        VectorOperations.vectorToVectorScalarMultiplication(costFunctionDerivative, outputBatchSizeBuffer,
                costFunctionDerivative, outputSize * miniBatchSize);

        //dL/dw[n] = dL/dz[n] * a[n-1]^T
        calculateWeightsDelta(input, 0, costFunctionDerivative, calculatedWeightsDelta, calculatedBiasesDelta,
                miniBatchSize);
    }


    @Override
    public void backwardMiddleLayer(float[] input,
                                    float[] errors,
                                    float[] previousLayerActivationArgument,
                                    float[] weightsDelta, float[] biasesDelta,
                                    int miniBatchSize) {
        //y - output
        //dl/dz  - errors
        //[n] - current layer, [n-1] - previous layer
        //w - weights
        //b - biases
        //z = w * a + b
        //a[n-1] - input
        //a[n] - output

        //dL/dw[n] = dL/dz[n] * a[n-1]^T
        calculateWeightsDelta(input, 0, errors, weightsDelta, biasesDelta, miniBatchSize);

        //dL/dz[n-1] = w[n]^T * dL/dz[n] * g'(z[n-1])
        calculatePreviousLayerError(errors, previousLayerActivationArgument,
                miniBatchSize);
    }

    private void calculatePreviousLayerError(float[] currentLayerErrors,
                                             float[] previousLayerActivationArgument,
                                             int miniBatchSize) {
        var weightsSizeBuffer = getWeightsSizeBuffer(inputSize * outputSize);
        //w[n]^T
        MatrixOperations.transposeMatrix(weights, 0, outputSize, inputSize, weightsSizeBuffer);

        var firstOutputBatchSizeBuffer = getFirstOutputXBatchSizeBuffer(miniBatchSize * outputSize);
        //w[n]^T * dL/dz[n]
        MatrixOperations.matrixToMatrixMultiplication(weightsSizeBuffer, 0, inputSize, outputSize,
                currentLayerErrors, 0, outputSize, miniBatchSize, firstOutputBatchSizeBuffer);

        var inputBatchSizeBuffer = getInputXBatchSizeBuffer(miniBatchSize * inputSize);
        //g'(z[n-1])
        activationFunction.derivative(previousLayerActivationArgument, inputBatchSizeBuffer);

        //w[n]^T * dL/dz[n] * g'(z[n-1])
        VectorOperations.vectorToVectorScalarMultiplication(firstOutputBatchSizeBuffer, inputBatchSizeBuffer,
                currentLayerErrors, inputSize * miniBatchSize);
    }

    private void calculateWeightsDelta(float[] input, int inputOffset,
                                       float[] errors,
                                       float[] weightsDelta,
                                       float[] biasesDelta,
                                       int miniBatchSize) {
        var inputBatchSizeBuffer = getInputXBatchSizeBuffer(miniBatchSize * inputSize);
        //a[n-1]^T
        MatrixOperations.transposeMatrix(input, inputOffset, inputSize, miniBatchSize, inputBatchSizeBuffer);
        //dL/dw[n] = dL/dz[n] * a[n-1]^T
        MatrixOperations.matrixToMatrixMultiplication(errors, 0, outputSize, miniBatchSize,
                inputBatchSizeBuffer, 0, miniBatchSize, inputSize, weightsDelta);

        System.arraycopy(errors, 0, biasesDelta, 0, outputSize * miniBatchSize);
    }

    @Override
    public void backwardZeroLayer(float[] input, int inputOffset, float[] errors, float[] weightsDelta,
                                  float[] biasesDelta,
                                  int miniBatchSize) {
        //y - output
        //dl/dz  - errors
        //[n] - current layer, [n-1] - previous layer
        //w - weights
        //b - biases
        //z = w * a + b
        //a[n-1] - input
        //a[n] - output

        //dL/dw[n] = dL/dz[n] * a[n-1]^T
        calculateWeightsDelta(input, inputOffset, errors, weightsDelta, biasesDelta, miniBatchSize);
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
    public void updateWeightsAndBiases(float[] weightsDelta, float[] biasesDelta,
                                       float learningRate, int miniBatchSize) {
        var outputBatchSizeBuffer = getFirstOutputXBatchSizeBuffer(miniBatchSize * outputSize);
        //calculate average of the weights and biases deltas
        MatrixOperations.reduceMatrixToVector(biasesDelta, outputSize, miniBatchSize, outputBatchSizeBuffer);
        VectorOperations.multiplyVectorToScalar(outputBatchSizeBuffer, 0, 1.0f / miniBatchSize,
                outputBatchSizeBuffer, 0, outputSize);
        VectorOperations.multiplyVectorToScalar(weightsDelta, 0, 1.0f / miniBatchSize,
                weightsDelta, 0, inputSize * outputSize);

        optimizer.optimize(weights, weightsDelta, inputSize * outputSize, biases, outputBatchSizeBuffer,
                outputSize, learningRate);
    }

    @Override
    public void saveBestWeightsAndBiases() {
        bestWeights = weights.clone();
        bestBiases = biases.clone();
    }

    @Override
    public void restoreBestWeightsAndBiases() {
        System.arraycopy(bestWeights, 0, weights, 0, weights.length);
        System.arraycopy(bestBiases, 0, biases, 0, biases.length);
    }

    private static float[] getFirstOutputXBatchSizeBuffer(int size) {
        var buffer = firstOutputXBatchSizeBuffer.get();

        if (buffer == null || buffer.length < size) {
            buffer = new float[size];
            firstOutputXBatchSizeBuffer.set(buffer);
        }

        return buffer;
    }

    private static float[] getSecondOutputXBatchSizeBuffer(int size) {
        var buffer = secondOutputXBatchSizeBuffer.get();

        if (buffer == null || buffer.length < size) {
            buffer = new float[size];
            secondOutputXBatchSizeBuffer.set(buffer);
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

    private static float[] getInputXBatchSizeBuffer(int size) {
        var buffer = inputXBatchSizeBuffer.get();

        if (buffer == null || buffer.length < size) {
            buffer = new float[size];
            inputXBatchSizeBuffer.set(buffer);
        }

        return buffer;
    }
}
