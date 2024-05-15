package com.tornadoml.cpu;


import org.apache.commons.rng.sampling.PermutationSampler;
import org.apache.commons.rng.simple.RandomSource;

import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Supplier;
import java.util.stream.Stream;

public final class NeuralNetwork {
    private final Layer[] layers;

    private final CostFunction costFunction;


    public NeuralNetwork(CostFunction costFunction,
        Layer... layers) {
        this.layers = layers;
        this.costFunction = costFunction;
    }

    public float[] predict(float[] input) {
        var output = new float[layers[0].getOutputSize()];
        input = Arrays.copyOf(input, layers[0].getInputSize());

        for (int i = 0; i < layers.length; i++) {
            layers[i].predict(input, output, 1);
            if (i < layers.length - 1) {
                var tmp = input;

                input = output;
                if (output.length < layers[i].getOutputSize()) {
                    output = new float[layers[i].getOutputSize()];
                } else {
                    output = tmp;
                }
            }
        }

        return output;
    }

    public void train(Supplier<Stream<float[]>> inputDataSupplier,
                      Supplier<Stream<float[]>> targetDataSupplier,
                      int inputSize, int targetSize, int startBatchSize,
                      int maxBatchCapacity, int miniBatchSize, int maxEpochs,
                      float learningRate, int patience) throws Exception {
        var batchCapacity = startBatchSize;
        var batchSize = 0;

        float[] batchInput = new float[batchCapacity * inputSize];
        float[] batchTarget = new float[batchCapacity * targetSize];
        var cores = Runtime.getRuntime().availableProcessors();

        System.out.println("Cores: " + cores);
        var maxOutputSize = 0;
        var maxInputSize = 0;

        for (Layer layer : layers) {
            maxOutputSize = Math.max(maxOutputSize, layer.getOutputSize());
            maxInputSize = Math.max(maxInputSize, layer.getInputSize());
        }

        var bestCost = Float.MAX_VALUE;
        var patienceCounter = 0;

        try (var executor = Executors.newFixedThreadPool(cores)) {
            epochLoop:
            for (int n = 0; n < maxEpochs; n++) {
                var inputData = inputDataSupplier.get();
                var targetData = targetDataSupplier.get();

                var inputDataIterator = inputData.iterator();
                var targetDataIterator = targetData.iterator();

                while (inputDataIterator.hasNext()) {
                    batchSize = 0;

                    while (inputDataIterator.hasNext()) {
                        var inputDatum = inputDataIterator.next();
                        var targetDatum = targetDataIterator.next();

                        if (inputDatum.length != inputSize) {
                            throw new IllegalArgumentException("Invalid input data size");
                        }
                        if (targetDatum.length != targetSize) {
                            throw new IllegalArgumentException("Invalid target data size");
                        }

                        if (batchSize == batchCapacity) {
                            if (batchCapacity < maxBatchCapacity) {
                                batchCapacity = Math.min(maxBatchCapacity, batchCapacity << 1);

                                var newBatchInput = new float[batchCapacity * inputSize];
                                var newBatchTarget = new float[batchCapacity * targetSize];

                                System.arraycopy(batchInput, 0, newBatchInput, 0, batchInput.length);
                                System.arraycopy(batchTarget, 0, newBatchTarget, 0, batchTarget.length);

                                batchInput = newBatchInput;
                                batchTarget = newBatchTarget;
                            } else {
                                break;
                            }
                        }

                        System.arraycopy(inputDatum, 0, batchInput, batchSize * inputSize,
                                inputSize);
                        System.arraycopy(targetDatum, 0, batchTarget, batchSize * targetSize,
                                targetSize);

                        batchSize++;
                    }

                    var cost = trainingCost(layers, costFunction, maxOutputSize, batchSize, batchInput,
                            batchTarget, executor, cores);

                    if (bestCost < cost) {
                        patienceCounter++;

                        if (patienceCounter >= patience) {
                            System.out.println("Reached patience limit. Stopping training.");
                            break epochLoop;
                        }
                    } else {
                        bestCost = cost;
                        patienceCounter = 0;

                        for (Layer layer : layers) {
                            if (layer instanceof TrainableLayer trainableLayer) {
                                trainableLayer.saveBestWeightsAndBiases();
                            }
                        }
                    }

                    System.out.println("Epoch: " + n + " Cost: " + cost + " best cost: " + bestCost +
                            " patience counter: " + patienceCounter);
                    trainBatch(batchInput, batchTarget, batchSize, miniBatchSize, learningRate, executor, cores);
                }
            }

            for (Layer layer : layers) {
                if (layer instanceof TrainableLayer trainableLayer) {
                    trainableLayer.restoreBestWeightsAndBiases();
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

    private static float trainingCost(Layer[] layers, CostFunction costFunction,
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

                ((TrainableLayer) layers[0]).forwardTraining(batchInput, start, activationArguments,
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
                return costFunction.value(predictions, 0, batchTarget, start,
                        outputSize * submitSize, batchSize);
            });

            submitted++;
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
            input[n] = new float[layers[0].getInputSize() * miniBatchSize];
            target[n] = new float[layers[layers.length - 1].getOutputSize() * miniBatchSize];

            for (int i = 0; i < layers.length; i++) {
                var layer = layers[i];
                activationArguments[n][i] = new float[miniBatchSize * layer.getOutputSize()];
                predictions[n][i] = new float[miniBatchSize * layer.getOutputSize()];
                weightsDelta[n][i] = new float[layer.getInputSize() * layer.getOutputSize()];
                biasesDelta[n][i] = new float[miniBatchSize * layer.getOutputSize()];
            }
        }

        var costErrors = new float[cores][maxOutputSize * miniBatchSize];
        var shuffledIndices = PermutationSampler.natural(batchSize);
        PermutationSampler.shuffle(RandomSource.ISAAC.create(), shuffledIndices);

        var propagationFutures = new Future[cores];
        var submitedTasks = 0;

        for (int miniBatchIndex = 0; miniBatchIndex < miniBatchCount; ) {
            for (int inputIndex = 0;
                 submitedTasks < cores && miniBatchIndex < miniBatchCount; submitedTasks++, miniBatchIndex++,
                         inputIndex += miniBatchSize) {
                var threadIndex = submitedTasks;
                var localInputIndex = inputIndex;
                var submitSize = Math.min(miniBatchSize, batchSize - inputIndex);

                propagationFutures[submitedTasks] = executor.submit(() ->
                        singleMiniBatchCycle(layers, costFunction, batchInput, batchTarget, activationArguments[threadIndex],
                                predictions[threadIndex], costErrors[threadIndex],
                                weightsDelta[threadIndex], biasesDelta[threadIndex],
                                input[threadIndex], target[threadIndex], submitSize,
                                shuffledIndices, localInputIndex));
            }

            var weightsDeltaSum = weightsDelta[0];
            var biasesDeltaSum = biasesDelta[0];

            for (int t = 0; t < submitedTasks; t++) {
                var future = propagationFutures[t];
                future.get();

                if (t > 0) {
                    for (int n = 0; n < layers.length; n++) {
                        VectorOperations.addVectorToVector(biasesDeltaSum[n], biasesDelta[t][n], biasesDeltaSum[n],
                                miniBatchSize * layers[n].getOutputSize());
                    }

                    for (int n = 0; n < layers.length; n++) {
                        VectorOperations.addVectorToVector(weightsDeltaSum[n], weightsDelta[t][n], weightsDeltaSum[n],
                                layers[n].getOutputSize() * layers[n].getInputSize());
                    }
                }
            }

            for (int n = 0; n < layers.length; n++) {
                if (layers[n] instanceof TrainableLayer trainableLayer) {
                    trainableLayer.updateWeightsAndBiases(weightsDeltaSum[n], biasesDeltaSum[n],
                            learningRate, miniBatchSize);
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
                                             float[] input, float[] target,
                                             int submitSize, int[] shuffledIndices, int localInputIndex) {
        var inputSize = layers[0].getInputSize();
        var outputSize = layers[layers.length - 1].getOutputSize();
        var lastLayerIndex = layers.length - 1;

        for (int i = 0; i < submitSize; i++) {
            var shuffledIndex = shuffledIndices[localInputIndex + i];
            System.arraycopy(batchInput, shuffledIndex * inputSize,
                    input, i * inputSize, inputSize);
            System.arraycopy(batchTarget, shuffledIndex * outputSize,
                    target, i * outputSize, outputSize);
        }

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

        costFunction.derivative(predictions[lastLayerIndex], 0, target, 0,
                costErrors, 0, outputSize * submitSize);
        //backward step
        var lastLayer = layers[lastLayerIndex];
        if (lastLayer instanceof TrainableLayer trainableLayer) {
            trainableLayer.backwardLastLayer(input, activationArguments[lastLayerIndex - 1],
                    activationArguments[lastLayerIndex], costErrors, weightsDelta[lastLayerIndex],
                    biasesDelta[lastLayerIndex],
                    submitSize);
        } else {
            ((NonTrainableLayer)lastLayer).backwardLastLayer(predictions[lastLayerIndex - 1], target, costErrors, submitSize);
        }

        for (int n = lastLayerIndex - 1; n > 0; n--) {
            ((TrainableLayer)layers[n]).backwardMiddleLayer(predictions[n - 1], costErrors,
                    activationArguments[n - 1], weightsDelta[n], biasesDelta[n],
                    submitSize);
        }

        assert layers[0] instanceof TrainableLayer;
        ((TrainableLayer)layers[0]).backwardZeroLayer(input, 0, costErrors,
                weightsDelta[0], biasesDelta[0],
                submitSize);
    }
}
