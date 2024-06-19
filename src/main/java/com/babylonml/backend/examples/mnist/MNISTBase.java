package com.babylonml.backend.examples.mnist;

import com.babylonml.backend.training.execution.InputSource;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import com.babylonml.backend.training.initializer.Initializer;
import com.babylonml.backend.training.operations.*;
import com.babylonml.backend.training.optimizer.AMSGradOptimizer;

public abstract class MNISTBase {
    @SuppressWarnings("SameParameterValue")
    protected static void trainMnist(int... neuronsCount) throws Exception {
        var inputSize = 784;
        var outputSize = 10;

        var miniBatchSize = 1024;
        var learningRate = 0.001f;

        var trainingExecutionContext = new TrainingExecutionContext(10, miniBatchSize);
        var trainingImages = MNISTLoader.loadMNISTImages();
        var trainingLabels = MNISTLoader.loadMNISTLabels();

        var trainingLabelProbabilities = new float[trainingLabels.length][outputSize];
        for (int i = 0; i < trainingLabels.length; i++) {
            trainingLabelProbabilities[i][trainingLabels[i]] = 1.0f;
        }

        var inputSource = trainingExecutionContext.registerMainInputSource(trainingImages);
        var fnnOutput = createFNN(neuronsCount, inputSize, inputSource, trainingExecutionContext, learningRate, outputSize);

        var expectedValuesSource = trainingExecutionContext.registerAdditionalInputSource(trainingLabelProbabilities);
        trainingExecutionContext.initializeExecution(new CrossEntropyCostFunction(expectedValuesSource, fnnOutput));

        trainingExecutionContext.executePropagation(((epochIndex, result) ->
                System.out.println("Epoch: " + epochIndex + ", loss: " + result)));

//        var trainingAccuracy = network.test(trainingImages, trainingLabelProbabilities);
//        System.out.println("Training accuracy: " + trainingAccuracy);
    }

    private static Operation createFNN(int[] neuronsCount, int inputSize, InputSource inputSource,
                                       TrainingExecutionContext trainingExecutionContext,
                                       float learningRate, int outputSize) {
        var layer = denseLayer(inputSize, neuronsCount[0], inputSource, inputSource,
                trainingExecutionContext, 0, learningRate);

        for (int i = 1; i < neuronsCount.length; i++) {
            layer = denseLayer(neuronsCount[i - 1], neuronsCount[i], inputSource, layer,
                    trainingExecutionContext, i, learningRate);
        }
        return denseLayer(neuronsCount[neuronsCount.length - 1], outputSize, inputSource,
                layer, trainingExecutionContext, neuronsCount.length, learningRate);
    }

    private static Operation denseLayer(int inputSize, int outputSize, InputSource inputSource,
                                        Operation input,
                                        TrainingExecutionContext executionContext, int layerIndex,
                                        float learningRate) {
        var weights = new Variable("weights" + layerIndex, executionContext,
                new AMSGradOptimizer(inputSource), new int[]{inputSize, outputSize},
                learningRate, Initializer.he());
        var biases = new Variable("biases" + layerIndex, executionContext,
                new AMSGradOptimizer(inputSource), new int[]{1, outputSize},
                learningRate, Initializer.he());

        var linear = new Add(new Multiplication(input, weights), biases);

        return new GeLUFunction(linear);
    }
}
