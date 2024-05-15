package com.tornadoml.mnist;

import com.tornadoml.cpu.*;

import java.util.Arrays;

import com.tornadoml.cpu.WeightsOptimizer.OptimizerType;

public abstract class MNISTBase {
    @SuppressWarnings("SameParameterValue")
    protected static void trainMnist(int maxEpochs, int... neuronsCount) throws Exception {
        var inputSize = 784;
        var outputSize = 10;

        var miniBatchSize = 1024;
        var learningRate = 0.001f;

        Layer[] layers = new Layer[neuronsCount.length + 2];
        layers[0] = new DenseLayer(inputSize, neuronsCount[0], new GeLU(), OptimizerType.AMS_GRAD);
        for (int i = 1; i < neuronsCount.length; i++) {
            layers[i] = new DenseLayer(neuronsCount[i - 1], neuronsCount[i], new GeLU(), OptimizerType.AMS_GRAD);
        }
        layers[neuronsCount.length] = new DenseLayer(neuronsCount[neuronsCount.length - 1], outputSize, new GeLU(),
                OptimizerType.AMS_GRAD);
        layers[neuronsCount.length + 1] = new SoftMaxLayer(outputSize);

        var network = new NeuralNetwork(new CategoricalCrossEntropyFunction(), layers);

        var trainingImages = MNISTLoader.loadMNISTImages();
        var trainingLabels = MNISTLoader.loadMNISTLabels();

        var trainingLabelProbabilities = new float[trainingLabels.length][outputSize];
        for (int i = 0; i < trainingLabels.length; i++) {
            trainingLabelProbabilities[i][trainingLabels[i]] = 1.0f;
        }

        network.train(() -> Arrays.stream(trainingImages), () -> Arrays.stream(trainingLabelProbabilities),
                //frozen
                inputSize, outputSize, 100, Integer.MAX_VALUE,

                //variable
                miniBatchSize,
                maxEpochs,
                learningRate, 50);

        var testImages = MNISTLoader.loadMNISTTestImages();
        var testLabels = MNISTLoader.loadMNISTTestLabels();

        var testLabelProbabilities = new float[testLabels.length][outputSize];
        for (int i = 0; i < testLabels.length; i++) {
            testLabelProbabilities[i][testLabels[i]] = 1.0f;
        }

        var testAccuracy = network.test(testImages, testLabelProbabilities);
        System.out.println("Test accuracy: " + testAccuracy);

        var trainingAccuracy = network.test(trainingImages, trainingLabelProbabilities);
        System.out.println("Training accuracy: " + trainingAccuracy);

    }
}
