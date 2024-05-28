package com.tornadoml.cpu;


import org.apache.commons.rng.sampling.PermutationSampler;
import org.apache.commons.rng.simple.RandomSource;

import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public final class NeuralNetwork {
    private static final float DEFAULT_COST_THRESHOLD = 0.001f; //0.1%
    private static final int DEFAULT_EPOCHS = 100;

    private final Layer[] layers;

    private final CostFunction costFunction;
    private final int cores;


    public NeuralNetwork(CostFunction costFunction,
                         Layer... layers) {
        this(costFunction, -1, layers);
    }

    public NeuralNetwork(CostFunction costFunction, int cores,
                         Layer... layers) {
        this.layers = layers;
        this.costFunction = costFunction;
        if (cores > 0) {
            this.cores = cores;
        } else {
            this.cores = Runtime.getRuntime().availableProcessors();
        }
    }

    public float[] predict(float[] input) {
        var output = new float[layers[0].getOutputSize()];
        input = Arrays.copyOf(input, layers[0].getInputSize());

        for (int i = 0; i < layers.length; i++) {
            layers[i].predict(input, output, 1);
            if (i < layers.length - 1) {
                var tmp = input;

                input = output;
                var nextLayerOutputSize = layers[i + 1].getOutputSize();
                if (tmp.length < nextLayerOutputSize) {
                    output = new float[nextLayerOutputSize];
                } else {
                    output = tmp;
                }
            }
        }

        var lastLayer = layers[layers.length - 1];
        return Arrays.copyOf(output, lastLayer.getOutputSize());
    }

    public void fit(float[][] inputData,
                    float[][] targetData,
                    int inputSize, int targetSize, int batchSize,
                    int miniBatchSize,
                    float learningRate, int patience, boolean shuffle) throws Exception {
        fit(inputData, targetData, inputSize, targetSize, batchSize, miniBatchSize, DEFAULT_EPOCHS, learningRate, patience,
                shuffle, DEFAULT_COST_THRESHOLD);
    }

    public void fit(float[][] inputData,
                    float[][] targetData,
                    int inputSize, int targetSize, int batchSize,
                    int miniBatchSize, int maxEpochs,
                    float learningRate, int patience, boolean shuffle, float thresholdLimit) throws Exception {
        System.out.println("Cores: " + cores);
        var maxOutputSize = 0;
        var maxInputSize = 0;

        for (Layer layer : layers) {
            maxOutputSize = Math.max(maxOutputSize, layer.getOutputSize());
            maxInputSize = Math.max(maxInputSize, layer.getInputSize());
        }

        var bestCost = Float.MAX_VALUE;
        var patienceCounter = 0;

        var batchInput = new float[batchSize * inputSize];
        var batchTarget = new float[batchSize * targetSize];

        var shuffledIndices = PermutationSampler.natural(batchSize);
        if (shuffle) {
            PermutationSampler.shuffle(RandomSource.ISAAC.create(), shuffledIndices);
        }

        var transposeBuffer = new float[batchSize * Math.max(inputSize, targetSize)];
        for (int i = 0; i < batchSize; i++) {
            var index = shuffledIndices[i];
            System.arraycopy(inputData[index], 0, transposeBuffer, i * inputSize, inputSize);
        }
        MatrixOperations.transposeMatrix(transposeBuffer, 0, batchSize, inputSize, batchInput);

        for (int i = 0; i < batchSize; i++) {
            var index = shuffledIndices[i];
            System.arraycopy(targetData[index], 0, transposeBuffer, i * targetSize, targetSize);
        }
        MatrixOperations.transposeMatrix(transposeBuffer, 0, batchSize, targetSize, batchTarget);


        try (var executor = Executors.newFixedThreadPool(cores)) {
            for (int n = 0; n < maxEpochs; n++) {
                var cost = trainingCost(layers, costFunction, maxOutputSize, inputData.length, batchInput,
                        batchTarget, executor, cores);
                float threshold = Float.NaN;


                if (!Float.isFinite(cost)) {
                    System.out.println("Cost is not finite, stopping fitting.");
                    break;
                }

                if (bestCost < cost) {
                    if (patience > -1) {
                        patienceCounter++;

                        if (patienceCounter >= patience) {
                            System.out.println("Reached patience limit. Stopping fitting.");
                        }
                    }
                } else {
                    threshold = Math.abs(cost - bestCost) / bestCost;
                    bestCost = cost;
                    if (patience > -1) {
                        for (Layer layer : layers) {
                            if (layer instanceof TrainableLayer trainableLayer) {
                                trainableLayer.saveWeightsAndBiases();
                            }
                        }

                        if (threshold >= thresholdLimit) {
                            patienceCounter = 0;
                        }
                    }
                }

                if (Float.isFinite(threshold)) {
                    System.out.println("Epoch: " + n + " Cost: " + cost + " best cost: " + bestCost +
                            " patience counter: " + patienceCounter + " threshold: "
                            + threshold + " threshold limit " + thresholdLimit);
                } else {
                    System.out.println("Epoch: " + n + " Cost: " + cost + " best cost: " + bestCost +
                            " patience counter: " + patienceCounter);
                }

                trainBatch(batchInput, batchTarget, batchSize, miniBatchSize, learningRate, executor, cores);
            }

            if (patience > -1) {
                for (var layer : layers) {
                    if (layer instanceof TrainableLayer trainableLayer) {
                        trainableLayer.restoreBestWeightsAndBiases();
                    }
                }
            }


            var cost = trainingCost(layers, costFunction, maxOutputSize, batchSize, batchInput, batchTarget,
                    executor, cores);
            System.out.println("Final cost: " + cost);
        }
    }


    public Object test(float[][] input, float[][] target) {
        assert input.length == target.length;
        var accuracy = 0.0f;
        var outputSize = layers[layers.length - 1].getOutputSize();

        for (int i = 0; i < input.length; i++) {
            var prediction = predict(input[i]);

            var maxIndex = 0;
            var maxValue = prediction[0];

            for (int j = 1; j < outputSize; j++) {
                if (prediction[j] > maxValue) {
                    maxIndex = j;
                    maxValue = prediction[j];
                }
            }

            if (target[i][maxIndex] == 1.0f) {
                accuracy += 1.0f;
            }
        }

        return accuracy / input.length;
    }

    static float trainingCost(Layer[] layers, CostFunction costFunction,
                              int maxOutputSize, int batchSize, float[] batchInput, float[] batchTarget,
                              ExecutorService executor, int cores) {
        var futures = new Future[cores];
        var submitted = 0;

        var maxSubmitSize = (batchSize + cores - 1) / cores;
        for (int t = 0; t < cores; t++) {
            var start = t * maxSubmitSize;
            var end = Math.min(start + maxSubmitSize, batchSize);
            var submitSize = end - start;

            futures[t] = executor.submit(() -> {
                var activationArguments = new float[maxOutputSize * submitSize];
                var predictions = new float[maxOutputSize * submitSize];
                var input = new float[layers[0].getInputSize() * submitSize];
                var target = new float[maxOutputSize * submitSize];

                MatrixOperations.subMatrix(batchInput, start, layers[0].getInputSize(), batchSize,
                        input, submitSize);
                MatrixOperations.subMatrix(batchTarget, start, layers[layers.length - 1].getOutputSize(), batchSize,
                        target, submitSize);

                ((TrainableLayer) layers[0]).forwardTraining(input, 0, activationArguments,
                        predictions, submitSize);
                for (int i = 1; i < layers.length; i++) {
                    var layer = layers[i];
                    if (layer instanceof TrainableLayer trainableLayer) {
                        trainableLayer.forwardTraining(predictions, 0, activationArguments,
                                predictions, submitSize);
                    } else {
                        layer.predict(predictions, predictions, submitSize);
                    }
                }

                var outputSize = layers[layers.length - 1].getOutputSize();
                return costFunction.value(predictions, 0, target, 0,
                        outputSize * submitSize, batchSize);
            });

            submitted++;
            if (end >= batchSize) {
                break;
            }
        }

        var cost = 0f;
        for (int t = 0; t < submitted; t++) {
            try {
                cost += (Float) futures[t].get();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        return cost;
    }

    private void trainBatch(float[] batchInput, float[] batchTarget,
                            int batchSize, int miniBatchSize, float learningRate, ExecutorService executor,
                            int cores) throws Exception {
        int miniBatchCount = (batchSize + miniBatchSize - 1) / miniBatchSize;
        int miniBatchSizePerCore = (miniBatchSize + cores - 1) / cores;

        var maxOutputSize = 0;
        var maxInputSize = 0;

        for (Layer layer : layers) {
            maxOutputSize = Math.max(maxOutputSize, layer.getOutputSize());
            maxInputSize = Math.max(maxInputSize, layer.getInputSize());
        }

        var activationArguments = new float[cores][layers.length][];
        var predictions = new float[cores][layers.length][];
        var weightsDelta = new float[cores][layers.length][];
        var biasesDelta = new float[cores][layers.length][];

        var input = new float[cores][];
        var target = new float[cores][];

        for (int n = 0; n < cores; n++) {
            input[n] = new float[layers[0].getInputSize() * miniBatchSizePerCore];
            target[n] = new float[layers[layers.length - 1].getOutputSize() * miniBatchSizePerCore];

            for (int i = 0; i < layers.length; i++) {
                var layer = layers[i];
                activationArguments[n][i] = new float[miniBatchSizePerCore * layer.getOutputSize()];
                predictions[n][i] = new float[miniBatchSizePerCore * layer.getOutputSize()];
                weightsDelta[n][i] = new float[layer.getInputSize() * layer.getOutputSize()];
                biasesDelta[n][i] = new float[miniBatchSizePerCore * layer.getOutputSize()];
            }
        }

        var costErrors = new float[cores][maxOutputSize * miniBatchSizePerCore];

        var propagationFutures = new Future[cores];
        var submittedSizes = new int[cores];
        var submittedIndexes = 0;
        var submitedTasks = 0;

        for (int inputIndex = 0, miniBatchIndex = 0; miniBatchIndex < miniBatchCount; miniBatchIndex++) {
            for (submittedIndexes = 0; submitedTasks < cores && submittedIndexes < miniBatchSize; submitedTasks++) {
                var threadIndex = submitedTasks;

                var localInputIndex = inputIndex;
                var submitSize = Math.min(Math.min(miniBatchSizePerCore, batchSize - inputIndex), miniBatchSize - submittedIndexes);

                inputIndex += submitSize;
                submittedIndexes += submitSize;
                submittedSizes[threadIndex] = submitSize;

                propagationFutures[submitedTasks] = executor.submit(() ->
                        singleMiniBatchCycle(layers, costFunction, batchInput, batchTarget, activationArguments[threadIndex],
                                predictions[threadIndex], costErrors[threadIndex],
                                weightsDelta[threadIndex], biasesDelta[threadIndex],
                                input[threadIndex], target[threadIndex], submitSize,
                                localInputIndex, batchSize));
            }

            var weightsDeltaSum = weightsDelta[0];
            var biasesDeltaSum = biasesDelta[0];

            for (int t = 0; t < submitedTasks; t++) {
                var future = propagationFutures[t];
                future.get();

                if (t > 0) {
                    for (int n = 0; n < layers.length; n++) {
                        var outputSize = layers[n].getOutputSize();
                        var biasesDeltaLayer = biasesDelta[t][n];

                        MatrixOperations.reduceMatrixToVector(biasesDeltaLayer, outputSize, submittedSizes[t],
                                biasesDeltaLayer);
                        VectorOperations.addVectorToVector(biasesDeltaSum[n], biasesDeltaLayer, biasesDeltaSum[n],
                                layers[n].getOutputSize());
                    }

                    for (int n = 0; n < layers.length; n++) {
                        VectorOperations.addVectorToVector(weightsDeltaSum[n], weightsDelta[t][n], weightsDeltaSum[n],
                                layers[n].getOutputSize() * layers[n].getInputSize());
                    }
                } else {
                    for (int n = 0; n < layers.length; n++) {
                        var outputSize = layers[n].getOutputSize();

                        MatrixOperations.reduceMatrixToVector(biasesDeltaSum[n], outputSize,
                                submittedSizes[0], biasesDeltaSum[n]);
                    }
                }
            }

            assert submittedIndexes > 0;
            for (int n = 0; n < layers.length; n++) {
                if (layers[n] instanceof TrainableLayer trainableLayer) {
                    //calculate average of the weights and biases deltas
                    var inputSize = layers[n].getInputSize();
                    var outputSize = layers[n].getOutputSize();

                    var biasesDeltaLayer = biasesDeltaSum[n];

                    VectorOperations.multiplyVectorToScalar(biasesDeltaLayer, 0, 1.0f / submittedIndexes,
                            biasesDeltaLayer, 0, outputSize);
                    VectorOperations.multiplyVectorToScalar(weightsDeltaSum[n], 0, 1.0f / submittedIndexes,
                            weightsDeltaSum[n], 0, inputSize * outputSize);

                    trainableLayer.updateWeightsAndBiases(weightsDeltaSum[n], biasesDeltaSum[n],
                            learningRate);
                }
            }

            submitedTasks = 0;
        }
    }

    private static void singleMiniBatchCycle(Layer[] layers, CostFunction costFunction,
                                             float[] batchInput, float[] batchTarget, float[][] activationArguments,
                                             float[][] predictions,
                                             float[] costErrors, float[][] weightsDelta,
                                             float[][] biasesDelta,
                                             float[] input, float[] expected,
                                             int submitSize, int localInputIndex, int batchSize) {
        var inputSize = layers[0].getInputSize();
        var outputSize = layers[layers.length - 1].getOutputSize();
        var lastLayerIndex = layers.length - 1;

        MatrixOperations.subMatrix(batchInput, localInputIndex, inputSize, batchSize,
                input, submitSize);
        MatrixOperations.subMatrix(batchTarget, localInputIndex, outputSize, batchSize,
                expected, submitSize);

        ((TrainableLayer) layers[0]).forwardTraining(input, 0, activationArguments[0],
                predictions[0], submitSize);

        for (int n = 1; n < layers.length; n++) {
            var layer = layers[n];
            if (layer instanceof TrainableLayer trainableLayer) {
                trainableLayer.forwardTraining(predictions[n - 1], 0,
                        activationArguments[n], predictions[n], submitSize);
            } else {
                layer.predict(predictions[n - 1], predictions[n], submitSize);
            }
        }

        costFunction.derivative(predictions[lastLayerIndex], 0, expected, 0,
                costErrors, 0, outputSize * submitSize);
        if (lastLayerIndex > 0) {
            //backward step
            var lastLayer = layers[lastLayerIndex];
            if (lastLayer instanceof TrainableLayer trainableLayer) {
                trainableLayer.backwardLastLayer(predictions[lastLayerIndex - 1], activationArguments[lastLayerIndex - 1],
                        activationArguments[lastLayerIndex], costErrors, costErrors,
                        weightsDelta[lastLayerIndex],
                        biasesDelta[lastLayerIndex], submitSize);
            } else {
                ((NonTrainableLayer) lastLayer).backwardLastLayer(predictions[lastLayerIndex], expected, costErrors, submitSize);
            }

            for (int n = lastLayerIndex - 1; n > 0; n--) {
                ((TrainableLayer) layers[n]).backwardMiddleLayer(predictions[n - 1], costErrors,
                        activationArguments[n - 1], costErrors, weightsDelta[n],
                        biasesDelta[n], submitSize);
            }

            assert layers[0] instanceof TrainableLayer;
            ((TrainableLayer) layers[0]).backwardLastLayer(input, 0, costErrors,
                    weightsDelta[0], biasesDelta[0],
                    submitSize);
        } else {
            assert layers[0] instanceof TrainableLayer;
            ((TrainableLayer) layers[0]).backwardSingleLayerNoError(input, activationArguments[0],
                    costErrors, weightsDelta[0], biasesDelta[0],
                    submitSize);
        }
    }
}
