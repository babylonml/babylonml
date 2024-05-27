@file:Suppress("UnnecessaryVariable")

package com.tornadoml.cpu

import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class NeuralNetworkMSETests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun singleLayerSingleSampleTestOneEpoch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val input = FloatMatrix(inputSize, 1)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, 1)
        expected.fillRandom(source)

        val cores = source.nextInt(1, 10)
        val layer =
            DenseLayer(inputSize, outputSize, LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            layer
        )

        var weights = FloatMatrix(outputSize, inputSize, layer.weights)
        var biases = FloatMatrix(outputSize, 1, layer.biases)

        val z = weights * input + biases
        val prediction = leakyLeRU(z, 0.01f)

        val costError = mseCostFunctionDerivative(prediction, expected)
        val layerError = costError.hadamardMul(leakyLeRUDerivative(z, 0.01f))

        val weightsDelta = layerError * input.transpose()
        val biasesDelta = layerError

        val alpha = 0.001f

        weights -= weightsDelta * alpha
        biases -= biasesDelta * alpha

        neuralNetwork.fit(
            input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize, 1, 1,
            1, alpha, -1, false, Float.MIN_VALUE
        )

        Assertions.assertArrayEquals(weights.toFlatArray(), layer.weights, 0.0001f)
        Assertions.assertArrayEquals(biases.toFlatArray(), layer.biases, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun singleLayerPredictTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val input = FloatVector(inputSize)
        input.fillRandom(source)

        val cores = source.nextInt(1, 10)
        val layer =
            DenseLayer(inputSize, outputSize, LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            layer
        )

        val weights = FloatMatrix(outputSize, inputSize, layer.weights)
        val biases = FloatVector(layer.biases)

        val z = weights * input.broadcast(1) + biases.broadcast(1)
        val prediction = leakyLeRU(z, 0.01f)

        val result = neuralNetwork.predict(input.toArray())

        Assertions.assertArrayEquals(prediction.toFlatArray(), result, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun singleLayerMultiSampleTestOneEpoch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val sampleSize = source.nextInt(2, 50)

        val inputSize = source.nextInt(sampleSize, 100)
        val outputSize = source.nextInt(sampleSize, 100)
        val cores = source.nextInt(1, 10)

        val input = FloatMatrix(inputSize, sampleSize)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, sampleSize)
        expected.fillRandom(source)

        val layer =
            DenseLayer(inputSize, outputSize, LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(),
            cores,
            layer
        )

        var weights = FloatMatrix(outputSize, inputSize, layer.weights)
        var biases = FloatVector(layer.biases)

        val z = weights * input + biases.broadcast(sampleSize)

        val prediction = leakyLeRU(z, 0.01f)

        val costError = mseCostFunctionDerivative(prediction, expected)
        val layerError = costError.hadamardMul(leakyLeRUDerivative(z, 0.01f))

        val weightsDelta = layerError * input.transpose()

        val biasesDelta = layerError

        val alpha = 0.001f

        weights -= weightsDelta * alpha / sampleSize
        biases -= biasesDelta.reduce() * alpha / sampleSize

        neuralNetwork.fit(
            input.transpose().toArray(),
            expected.transpose().toArray(),
            inputSize,
            outputSize,
            sampleSize,
            sampleSize,
            1,
            alpha,
            -1,
            false, Float.MIN_VALUE
        )

        Assertions.assertArrayEquals(weights.toFlatArray(), layer.weights, 0.0001f)
        Assertions.assertArrayEquals(biases.toArray(), layer.biases, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun singleLayerMultiSampleTestSeveralEpochs(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val sampleSize = source.nextInt(2, 50)
        val epochs = source.nextInt(5, 50)

        val inputSize = source.nextInt(sampleSize, 100)
        val outputSize = source.nextInt(sampleSize, 100)

        val input = FloatMatrix(inputSize, sampleSize)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, sampleSize)
        expected.fillRandom(source)

        val cores = source.nextInt(1, 10)
        val layer =
            DenseLayer(inputSize, outputSize, LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            layer
        )

        var weights = FloatMatrix(outputSize, inputSize, layer.weights)
        var biases = FloatVector(layer.biases)
        val alpha = 0.001f

        for (epoch in 0 until epochs) {
            val z = weights * input + biases.broadcast(sampleSize)
            val prediction = leakyLeRU(z, 0.01f)

            val costError = mseCostFunctionDerivative(prediction, expected)
            val layerError = costError.hadamardMul(leakyLeRUDerivative(z, 0.01f))

            val weightsDelta = layerError * input.transpose()

            val biasesDelta = layerError

            weights -= weightsDelta * alpha / sampleSize
            biases -= biasesDelta.reduce() / sampleSize * alpha
        }

        neuralNetwork.fit(
            input.transpose().toArray(),
            expected.transpose().toArray(),
            inputSize,
            outputSize,
            sampleSize,
            sampleSize,
            epochs,
            alpha,
            -1,
            false, Float.MIN_VALUE
        )

        Assertions.assertArrayEquals(weights.toFlatArray(), layer.weights, 0.0001f)
        Assertions.assertArrayEquals(biases.toArray(), layer.biases, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun singleLayerMultiSampleTestSeveralEpochsMiniBatch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val sampleSize = source.nextInt(10, 100)
        val miniBatchSize = sampleSize / 10

        val epochs = source.nextInt(5, 50)

        val inputSize = source.nextInt(sampleSize, 100)
        val outputSize = source.nextInt(sampleSize, 100)

        val input = FloatMatrix(inputSize, sampleSize)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, sampleSize)
        expected.fillRandom(source)

        val cores = source.nextInt(1, 10)
        val layer =
            DenseLayer(inputSize, outputSize, LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            layer
        )

        var weights = FloatMatrix(outputSize, inputSize, layer.weights)
        var biases = FloatVector(layer.biases)

        val alpha = 0.001f
        val leakyLeRUGradient = 0.01f

        for (epoch in 0 until epochs) {
            for (i in 0 until sampleSize step miniBatchSize) {
                val miniSampleSize = min(miniBatchSize, sampleSize - i)

                val z = weights * input.subMatrix(i, miniSampleSize) + biases.broadcast(miniSampleSize)
                val prediction = leakyLeRU(z, leakyLeRUGradient)

                val costError = mseCostFunctionDerivative(prediction, expected.subMatrix(i, miniSampleSize))
                val layerError = costError.hadamardMul(leakyLeRUDerivative(z, leakyLeRUGradient))

                val weightsDelta = layerError * input.subMatrix(i, miniSampleSize).transpose()
                val biasesDelta = layerError

                weights -= weightsDelta * alpha / miniSampleSize
                biases -= biasesDelta.reduce() / miniSampleSize * alpha
            }
        }

        neuralNetwork.fit(
            input.transpose().toArray(),
            expected.transpose().toArray(),
            inputSize,
            outputSize,
            sampleSize,
            miniBatchSize,
            epochs,
            alpha,
            -1,
            false, Float.MIN_VALUE
        )

        Assertions.assertArrayEquals(weights.toFlatArray(), layer.weights, 0.0001f)
        Assertions.assertArrayEquals(biases.toArray(), layer.biases, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun singleLayerTrainingCost(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)
        val sampleSize = source.nextInt(1, 50)

        val input = FloatMatrix(inputSize, sampleSize)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, sampleSize)
        expected.fillRandom(source)

        val layer =
            DenseLayer(inputSize, outputSize, LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE)


        val weights = FloatMatrix(outputSize, inputSize, layer.weights)
        val biases = FloatVector(layer.biases)

        val z = weights * input + biases.broadcast(sampleSize)
        val prediction = leakyLeRU(z, 0.01f)

        val expectedCost = mseCostFunction(prediction, expected)

        val cores = source.nextInt(1, 10)
        val cost = Executors.newFixedThreadPool(cores).use { executor ->
            NeuralNetwork.trainingCost(
                arrayOf(layer), MSECostFunction(), outputSize, sampleSize, input.toFlatArray(), expected.toFlatArray(),
                executor, cores
            )
        }

        Assertions.assertEquals(expectedCost, cost, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun twoLayersSingleSampleTestOneEpoch(seed: Long, firstLayerSeed: Long, secondLayerSeed: Long) {
        val alpha = 0.001f
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val secondLayerSize = source.nextInt(10, 100)

        val input = FloatMatrix(inputSize, 1)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, 1)
        expected.fillRandom(source)

        val firstLayer =
            DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
        val secondLayer =
            DenseLayer(
                secondLayerSize,
                outputSize,
                LeakyLeRU(secondLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )

        val cores = source.nextInt(1, 10)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            firstLayer, secondLayer
        )

        var firstLayerWeights = FloatMatrix(secondLayerSize, inputSize, firstLayer.weights)
        var firstLayerBiases = FloatMatrix(secondLayerSize, 1, firstLayer.biases)

        var secondLayerWeights = FloatMatrix(outputSize, secondLayerSize, secondLayer.weights)
        var secondLayerBiases = FloatMatrix(outputSize, 1, secondLayer.biases)

        val firstZ = firstLayerWeights * input + firstLayerBiases
        val firstPrediction = leakyLeRU(firstZ, 0.01f)

        val secondZ = secondLayerWeights * firstPrediction + secondLayerBiases
        val secondPrediction = leakyLeRU(secondZ, 0.01f)

        val secondLayerCostError = mseCostFunctionDerivative(secondPrediction, expected)
        val secondLayerError = secondLayerCostError.hadamardMul(leakyLeRUDerivative(secondZ, 0.01f))

        val firstLayerError = (secondLayerWeights.transpose() * secondLayerError).hadamardMul(
            leakyLeRUDerivative(
                firstZ,
                0.01f
            )
        )

        val firstLayerWeightsDelta = firstLayerError * input.transpose()
        val firstLayerBiasesDelta = firstLayerError

        firstLayerWeights -= firstLayerWeightsDelta * alpha
        firstLayerBiases -= firstLayerBiasesDelta * alpha

        val secondLayerWeightsDelta = secondLayerError * firstPrediction.transpose()
        val secondLayerBiasesDelta = secondLayerError

        secondLayerWeights -= secondLayerWeightsDelta * alpha
        secondLayerBiases -= secondLayerBiasesDelta * alpha


        neuralNetwork.fit(
            input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize, 1, 1,
            1, alpha, -1, false, Float.MIN_VALUE
        )

        Assertions.assertArrayEquals(firstLayerWeights.toFlatArray(), firstLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(firstLayerBiases.toFlatArray(), firstLayer.biases, 0.0001f)

        Assertions.assertArrayEquals(secondLayerWeights.toFlatArray(), secondLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(secondLayerBiases.toFlatArray(), secondLayer.biases, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun twoLayerPredictTest(seed: Long, firstLayerSeed: Long, secondLayerSeed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val secondLayerSize = source.nextInt(10, 100)

        val input = FloatMatrix(inputSize, 1)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, 1)
        expected.fillRandom(source)

        val firstLayer =
            DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
        val secondLayer =
            DenseLayer(
                secondLayerSize,
                outputSize,
                LeakyLeRU(secondLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )

        val firstLayerWeights = FloatMatrix(secondLayerSize, inputSize, firstLayer.weights)
        val firstLayerBiases = FloatMatrix(secondLayerSize, 1, firstLayer.biases)

        val secondLayerWeights = FloatMatrix(outputSize, secondLayerSize, secondLayer.weights)
        val secondLayerBiases = FloatMatrix(outputSize, 1, secondLayer.biases)

        val firstZ = firstLayerWeights * input + firstLayerBiases
        val firstPrediction = leakyLeRU(firstZ, 0.01f)

        val secondZ = secondLayerWeights * firstPrediction + secondLayerBiases
        val prediction = leakyLeRU(secondZ, 0.01f)

        val neuralNetwork = NeuralNetwork(
            MSECostFunction(),
            firstLayer, secondLayer
        )

        val result = neuralNetwork.predict(input.toFlatArray())
        Assertions.assertArrayEquals(prediction.toFlatArray(), result, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun twoLayersSeveralSamplesTestOneEpoch(seed: Long, firstLayerSeed: Long, secondLayerSeed: Long) {
        val alpha = 0.001f
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)
        val samplesCount = source.nextInt(2, 50)

        val secondLayerSize = source.nextInt(10, 100)

        val input = FloatMatrix(inputSize, samplesCount)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, samplesCount)
        expected.fillRandom(source)

        val firstLayer =
            DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
        val secondLayer =
            DenseLayer(
                secondLayerSize,
                outputSize,
                LeakyLeRU(secondLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )

        val cores = source.nextInt(1, 10)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            firstLayer, secondLayer
        )

        var firstLayerWeights = FloatMatrix(secondLayerSize, inputSize, firstLayer.weights)
        var firstLayerBiases = FloatVector(firstLayer.biases)

        var secondLayerWeights = FloatMatrix(outputSize, secondLayerSize, secondLayer.weights)
        var secondLayerBiases = FloatVector(secondLayer.biases)

        val firstZ = firstLayerWeights * input + firstLayerBiases.broadcast(samplesCount)
        val firstPrediction = leakyLeRU(firstZ, 0.01f)

        val secondZ = secondLayerWeights * firstPrediction + secondLayerBiases.broadcast(samplesCount)
        val secondPrediction = leakyLeRU(secondZ, 0.01f)

        val secondLayerCostError = mseCostFunctionDerivative(secondPrediction, expected)
        val secondLayerError = secondLayerCostError.hadamardMul(leakyLeRUDerivative(secondZ, 0.01f))

        val firstLayerError = (secondLayerWeights.transpose() * secondLayerError).hadamardMul(
            leakyLeRUDerivative(
                firstZ,
                0.01f
            )
        )

        val firstLayerWeightsDelta = firstLayerError * input.transpose()
        val firstLayerBiasesDelta = firstLayerError

        firstLayerWeights -= firstLayerWeightsDelta * alpha / samplesCount
        firstLayerBiases -= firstLayerBiasesDelta.reduce() * alpha / samplesCount

        val secondLayerWeightsDelta = secondLayerError * firstPrediction.transpose()
        val secondLayerBiasesDelta = secondLayerError

        secondLayerWeights -= secondLayerWeightsDelta * alpha / samplesCount
        secondLayerBiases -= secondLayerBiasesDelta.reduce() * alpha / samplesCount


        neuralNetwork.fit(
            input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize, samplesCount,
            samplesCount, 1, alpha, -1, false, Float.MIN_VALUE
        )

        Assertions.assertArrayEquals(firstLayerWeights.toFlatArray(), firstLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(firstLayerBiases.toArray(), firstLayer.biases, 0.0001f)

        Assertions.assertArrayEquals(secondLayerWeights.toFlatArray(), secondLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(secondLayerBiases.toArray(), secondLayer.biases, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun twoLayersSeveralSamplesTestSeveralEpochs(seed: Long, firstLayerSeed: Long, secondLayerSeed: Long) {
        val alpha = 0.001f
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)
        val samplesCount = source.nextInt(2, 50)

        val secondLayerSize = source.nextInt(10, 100)

        val input = FloatMatrix(inputSize, samplesCount)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, samplesCount)
        expected.fillRandom(source)
        val firstLayer =
            DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
        val secondLayer =
            DenseLayer(
                secondLayerSize,
                outputSize,
                LeakyLeRU(secondLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )

        val cores = source.nextInt(1, 10)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            firstLayer, secondLayer
        )

        val epochs = source.nextInt(5, 50)
        var firstLayerWeights = FloatMatrix(secondLayerSize, inputSize, firstLayer.weights)
        var firstLayerBiases = FloatVector(firstLayer.biases)

        var secondLayerWeights = FloatMatrix(outputSize, secondLayerSize, secondLayer.weights)
        var secondLayerBiases = FloatVector(secondLayer.biases)

        for (epoch in 0 until epochs) {
            val firstZ = firstLayerWeights * input + firstLayerBiases.broadcast(samplesCount)
            val firstPrediction = leakyLeRU(firstZ, 0.01f)

            val secondZ = secondLayerWeights * firstPrediction + secondLayerBiases.broadcast(samplesCount)
            val secondPrediction = leakyLeRU(secondZ, 0.01f)

            val secondLayerCostError = mseCostFunctionDerivative(secondPrediction, expected)
            val secondLayerError = secondLayerCostError.hadamardMul(leakyLeRUDerivative(secondZ, 0.01f))

            val firstLayerError = (secondLayerWeights.transpose() * secondLayerError).hadamardMul(
                leakyLeRUDerivative(
                    firstZ,
                    0.01f
                )
            )

            val firstLayerWeightsDelta = firstLayerError * input.transpose()
            val firstLayerBiasesDelta = firstLayerError

            firstLayerWeights -= firstLayerWeightsDelta * alpha / samplesCount
            firstLayerBiases -= firstLayerBiasesDelta.reduce() * alpha / samplesCount

            val secondLayerWeightsDelta = secondLayerError * firstPrediction.transpose()
            val secondLayerBiasesDelta = secondLayerError

            secondLayerWeights -= secondLayerWeightsDelta * alpha / samplesCount
            secondLayerBiases -= secondLayerBiasesDelta.reduce() * alpha / samplesCount
        }

        neuralNetwork.fit(
            input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize, samplesCount,
            samplesCount, epochs, alpha, -1, false, Float.MIN_VALUE
        )

        Assertions.assertArrayEquals(firstLayerWeights.toFlatArray(), firstLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(firstLayerBiases.toArray(), firstLayer.biases, 0.0001f)

        Assertions.assertArrayEquals(secondLayerWeights.toFlatArray(), secondLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(secondLayerBiases.toArray(), secondLayer.biases, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun twoLayerMultiSampleTestSeveralEpochsMiniBatch(seed: Long, firstLayerSeed: Long, secondLayerSeed: Long) {
        val alpha = 0.001f
        val leakyLeRUGradient = 0.01f
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)
        val samplesCount = source.nextInt(10, 100)
        val miniBatchSize = samplesCount / 10

        val secondLayerSize = source.nextInt(10, 100)

        val input = FloatMatrix(inputSize, samplesCount)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, samplesCount)
        expected.fillRandom(source)
        val firstLayer =
            DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
        val secondLayer =
            DenseLayer(
                secondLayerSize,
                outputSize,
                LeakyLeRU(secondLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )

        val cores = source.nextInt(1, 10)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            firstLayer, secondLayer
        )

        val epochs = source.nextInt(5, 50)
        var firstLayerWeights = FloatMatrix(secondLayerSize, inputSize, firstLayer.weights)
        var firstLayerBiases = FloatVector(firstLayer.biases)

        var secondLayerWeights = FloatMatrix(outputSize, secondLayerSize, secondLayer.weights)
        var secondLayerBiases = FloatVector(secondLayer.biases)

        for (epoch in 0 until epochs) {
            for (i in 0 until samplesCount step miniBatchSize) {
                val miniSampleSize = min(miniBatchSize, samplesCount - i)
                val firstZ = firstLayerWeights * input.subMatrix(i, miniSampleSize) +
                        firstLayerBiases.broadcast(miniSampleSize)
                val firstPrediction = leakyLeRU(firstZ, 0.01f)

                val secondZ = secondLayerWeights * firstPrediction + secondLayerBiases.broadcast(miniSampleSize)
                val secondPrediction = leakyLeRU(secondZ, 0.01f)

                val secondLayerCostError = mseCostFunctionDerivative(
                    secondPrediction,
                    expected.subMatrix(i, miniSampleSize)
                )
                val secondLayerError = secondLayerCostError.hadamardMul(leakyLeRUDerivative(secondZ, leakyLeRUGradient))

                val firstLayerError = (secondLayerWeights.transpose() * secondLayerError).hadamardMul(
                    leakyLeRUDerivative(
                        firstZ,
                        leakyLeRUGradient
                    )
                )

                val firstLayerWeightsDelta = firstLayerError * input.subMatrix(i, miniSampleSize).transpose()
                val firstLayerBiasesDelta = firstLayerError

                firstLayerWeights -= firstLayerWeightsDelta * alpha / miniSampleSize
                firstLayerBiases -= firstLayerBiasesDelta.reduce() * alpha / miniSampleSize

                val secondLayerWeightsDelta = secondLayerError * firstPrediction.transpose()
                val secondLayerBiasesDelta = secondLayerError

                secondLayerWeights -= secondLayerWeightsDelta * alpha / miniSampleSize
                secondLayerBiases -= secondLayerBiasesDelta.reduce() * alpha / miniSampleSize
            }
        }

        neuralNetwork.fit(
            input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize, samplesCount,
            miniBatchSize, epochs, alpha, -1, false, Float.MIN_VALUE
        )

        Assertions.assertArrayEquals(firstLayerWeights.toFlatArray(), firstLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(firstLayerBiases.toArray(), firstLayer.biases, 0.0001f)

        Assertions.assertArrayEquals(secondLayerWeights.toFlatArray(), secondLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(secondLayerBiases.toArray(), secondLayer.biases, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun twoLayersTrainingCostTest(seed: Long, firstLayerSeed: Long, secondLayerSeed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val leakyLeRUGradient = 0.01f
        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)
        val samplesCount = source.nextInt(1, 50)

        val secondLayerSize = source.nextInt(10, 100)

        val input = FloatMatrix(inputSize, samplesCount)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, samplesCount)
        expected.fillRandom(source)

        val firstLayer =
            DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
        val secondLayer =
            DenseLayer(
                secondLayerSize,
                outputSize,
                LeakyLeRU(secondLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )

        val firstLayerWeights = FloatMatrix(secondLayerSize, inputSize, firstLayer.weights)
        val firstLayerBiases = FloatVector(firstLayer.biases)

        val secondLayerWeights = FloatMatrix(outputSize, secondLayerSize, secondLayer.weights)
        val secondLayerBiases = FloatVector(secondLayer.biases)

        val firstZ = firstLayerWeights * input + firstLayerBiases.broadcast(samplesCount)
        val firstPrediction = leakyLeRU(firstZ, leakyLeRUGradient)

        val secondZ = secondLayerWeights * firstPrediction + secondLayerBiases.broadcast(samplesCount)
        val secondPrediction = leakyLeRU(secondZ, leakyLeRUGradient)

        val expectedCost = mseCostFunction(secondPrediction, expected)

        val cores = source.nextInt(1, 10)
        val cost = Executors.newFixedThreadPool(cores).use { executor ->
            NeuralNetwork.trainingCost(
                arrayOf(firstLayer, secondLayer), MSECostFunction(),
                max(outputSize, secondLayerSize), samplesCount, input.toFlatArray(),
                expected.toFlatArray(), executor, cores
            )
        }

        Assertions.assertEquals(expectedCost, cost, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun threeLayersSingleSampleTestOneEpoch(
        seed: Long,
        firstLayerSeed: Long,
        secondLayerSeed: Long,
        thirdLayerSeed: Long
    ) {
        val alpha = 0.001f
        val leRUGradient = 0.01f

        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val secondLayerSize = source.nextInt(10, 100)
        val thirdLayerSize = source.nextInt(10, 100)

        val input = FloatMatrix(inputSize, 1)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, 1)
        expected.fillRandom(source)

        val firstLayer =
            DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
        val secondLayer =
            DenseLayer(
                secondLayerSize,
                thirdLayerSize,
                LeakyLeRU(secondLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )
        val thirdLayer =
            DenseLayer(
                thirdLayerSize,
                outputSize,
                LeakyLeRU(thirdLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )

        val cores = source.nextInt(1, 10)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            firstLayer, secondLayer, thirdLayer
        )

        var firstLayerWeights = FloatMatrix(secondLayerSize, inputSize, firstLayer.weights)
        var firstLayerBiases = FloatMatrix(secondLayerSize, 1, firstLayer.biases)

        var secondLayerWeights = FloatMatrix(thirdLayerSize, secondLayerSize, secondLayer.weights)
        var secondLayerBiases = FloatMatrix(thirdLayerSize, 1, secondLayer.biases)

        var thirdLayerWeights = FloatMatrix(outputSize, thirdLayerSize, thirdLayer.weights)
        var thirdLayerBiases = FloatMatrix(outputSize, 1, thirdLayer.biases)

        val firstZ = firstLayerWeights * input + firstLayerBiases
        val firstPrediction = leakyLeRU(firstZ, leRUGradient)

        val secondZ = secondLayerWeights * firstPrediction + secondLayerBiases
        val secondPrediction = leakyLeRU(secondZ, leRUGradient)

        val thirdZ = thirdLayerWeights * secondPrediction + thirdLayerBiases
        val thirdPrediction = leakyLeRU(thirdZ, leRUGradient)

        val thirdLayerCostError = mseCostFunctionDerivative(thirdPrediction, expected)
        val thirdLayerError = thirdLayerCostError.hadamardMul(leakyLeRUDerivative(thirdZ, leRUGradient))

        val secondLayerError = (thirdLayerWeights.transpose() * thirdLayerError).hadamardMul(
            leakyLeRUDerivative(
                secondZ,
                leRUGradient
            )
        )

        val firstLayerError = (secondLayerWeights.transpose() * secondLayerError).hadamardMul(
            leakyLeRUDerivative(
                firstZ,
                leRUGradient
            )
        )

        val firstLayerWeightsDelta = firstLayerError * input.transpose()
        val firstLayerBiasesDelta = firstLayerError

        firstLayerWeights -= firstLayerWeightsDelta * alpha
        firstLayerBiases -= firstLayerBiasesDelta * alpha

        val secondLayerWeightsDelta = secondLayerError * firstPrediction.transpose()
        val secondLayerBiasesDelta = secondLayerError

        secondLayerWeights -= secondLayerWeightsDelta * alpha
        secondLayerBiases -= secondLayerBiasesDelta * alpha

        val thirdLayerWeightsDelta = thirdLayerError * secondPrediction.transpose()
        val thirdLayerBiasesDelta = thirdLayerError

        thirdLayerWeights -= thirdLayerWeightsDelta * alpha
        thirdLayerBiases -= thirdLayerBiasesDelta * alpha

        neuralNetwork.fit(
            input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize,
            1, 1,
            1, alpha, -1, false, Float.MIN_VALUE
        )

        Assertions.assertArrayEquals(firstLayerWeights.toFlatArray(), firstLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(firstLayerBiases.toFlatArray(), firstLayer.biases, 0.0001f)

        Assertions.assertArrayEquals(secondLayerWeights.toFlatArray(), secondLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(secondLayerBiases.toFlatArray(), secondLayer.biases, 0.0001f)

        Assertions.assertArrayEquals(thirdLayerWeights.toFlatArray(), thirdLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(thirdLayerBiases.toFlatArray(), thirdLayer.biases, 0.0001f)

    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun threeLayerPredictTest(seed: Long, firstLayerSeed: Long, secondLayerSeed: Long,
                              thirdLayerSeed: Long) {
        val leRUGradient = 0.01f

        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val secondLayerSize = source.nextInt(10, 100)
        val thirdLayerSize = source.nextInt(10, 100)

        val input = FloatMatrix(inputSize, 1)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, 1)
        expected.fillRandom(source)

        val firstLayer =
            DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
        val secondLayer =
            DenseLayer(
                secondLayerSize,
                thirdLayerSize,
                LeakyLeRU(secondLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )
        val thirdLayer =
            DenseLayer(
                thirdLayerSize,
                outputSize,
                LeakyLeRU(thirdLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )

        val cores = source.nextInt(1, 10)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            firstLayer, secondLayer, thirdLayer
        )

        val firstLayerWeights = FloatMatrix(secondLayerSize, inputSize, firstLayer.weights)
        val firstLayerBiases = FloatMatrix(secondLayerSize, 1, firstLayer.biases)

        val secondLayerWeights = FloatMatrix(thirdLayerSize, secondLayerSize, secondLayer.weights)
        val secondLayerBiases = FloatMatrix(thirdLayerSize, 1, secondLayer.biases)

        val thirdLayerWeights = FloatMatrix(outputSize, thirdLayerSize, thirdLayer.weights)
        val thirdLayerBiases = FloatMatrix(outputSize, 1, thirdLayer.biases)

        val firstZ = firstLayerWeights * input + firstLayerBiases
        val firstPrediction = leakyLeRU(firstZ, leRUGradient)

        val secondZ = secondLayerWeights * firstPrediction + secondLayerBiases
        val secondPrediction = leakyLeRU(secondZ, leRUGradient)

        val thirdZ = thirdLayerWeights * secondPrediction + thirdLayerBiases
        val prediction = leakyLeRU(thirdZ, leRUGradient)

        val result = neuralNetwork.predict(input.toFlatArray())
        Assertions.assertArrayEquals(prediction.toFlatArray(), result, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun threeLayersMultiSampleTestOneEpoch(
        seed: Long,
        firstLayerSeed: Long,
        secondLayerSeed: Long,
        thirdLayerSeed: Long
    ) {

        val alpha = 0.001f
        val leRUGradient = 0.01f
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val secondLayerSize = source.nextInt(10, 100)
        val thirdLayerSize = source.nextInt(10, 100)

        val sampleCount = source.nextInt(2, 50)

        val input = FloatMatrix(inputSize, sampleCount)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, sampleCount)
        expected.fillRandom(source)

        val firstLayer =
            DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
        val secondLayer =
            DenseLayer(
                secondLayerSize,
                thirdLayerSize,
                LeakyLeRU(secondLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )
        val thirdLayer =
            DenseLayer(
                thirdLayerSize,
                outputSize,
                LeakyLeRU(thirdLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )

        val cores = source.nextInt(1, 10)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            firstLayer, secondLayer, thirdLayer
        )

        var firstLayerWeights = FloatMatrix(secondLayerSize, inputSize, firstLayer.weights)
        var firstLayerBiases = FloatVector(firstLayer.biases)

        var secondLayerWeights = FloatMatrix(thirdLayerSize, secondLayerSize, secondLayer.weights)
        var secondLayerBiases = FloatVector(secondLayer.biases)

        var thirdLayerWeights = FloatMatrix(outputSize, thirdLayerSize, thirdLayer.weights)
        var thirdLayerBiases = FloatVector(thirdLayer.biases)

        val firstZ = firstLayerWeights * input + firstLayerBiases.broadcast(sampleCount)
        val firstPrediction = leakyLeRU(firstZ, leRUGradient)

        val secondZ = secondLayerWeights * firstPrediction + secondLayerBiases.broadcast(sampleCount)
        val secondPrediction = leakyLeRU(secondZ, leRUGradient)

        val thirdZ = thirdLayerWeights * secondPrediction + thirdLayerBiases.broadcast(sampleCount)
        val thirdPrediction = leakyLeRU(thirdZ, leRUGradient)

        val thirdLayerCostError = mseCostFunctionDerivative(thirdPrediction, expected)
        val thirdLayerError = thirdLayerCostError.hadamardMul(leakyLeRUDerivative(thirdZ, leRUGradient))

        val secondLayerError = (thirdLayerWeights.transpose() * thirdLayerError).hadamardMul(
            leakyLeRUDerivative(
                secondZ,
                leRUGradient
            )
        )

        val firstLayerError = (secondLayerWeights.transpose() * secondLayerError).hadamardMul(
            leakyLeRUDerivative(
                firstZ,
                leRUGradient
            )
        )

        val firstLayerWeightsDelta = firstLayerError * input.transpose()
        val firstLayerBiasesDelta = firstLayerError

        firstLayerWeights -= firstLayerWeightsDelta * alpha / sampleCount
        firstLayerBiases -= firstLayerBiasesDelta.reduce() * alpha / sampleCount

        val secondLayerWeightsDelta = secondLayerError * firstPrediction.transpose()
        val secondLayerBiasesDelta = secondLayerError

        secondLayerWeights -= secondLayerWeightsDelta * alpha / sampleCount
        secondLayerBiases -= secondLayerBiasesDelta.reduce() * alpha / sampleCount

        val thirdLayerWeightsDelta = thirdLayerError * secondPrediction.transpose()
        val thirdLayerBiasesDelta = thirdLayerError

        thirdLayerWeights -= thirdLayerWeightsDelta * alpha / sampleCount
        thirdLayerBiases -= thirdLayerBiasesDelta.reduce() * alpha / sampleCount

        neuralNetwork.fit(
            input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize,
            sampleCount, sampleCount,
            1, alpha, -1, false, Float.MIN_VALUE
        )

        Assertions.assertArrayEquals(firstLayerWeights.toFlatArray(), firstLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(firstLayerBiases.toArray(), firstLayer.biases, 0.0001f)

        Assertions.assertArrayEquals(secondLayerWeights.toFlatArray(), secondLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(secondLayerBiases.toArray(), secondLayer.biases, 0.0001f)

        Assertions.assertArrayEquals(thirdLayerWeights.toFlatArray(), thirdLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(thirdLayerBiases.toArray(), thirdLayer.biases, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun threeLayersMultiSampleTestSeveralEpochs(
        seed: Long,
        firstLayerSeed: Long,
        secondLayerSeed: Long,
        thirdLayerSeed: Long
    ) {
        val alpha = 0.001f
        val leRUGradient = 0.01f

        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val secondLayerSize = source.nextInt(10, 100)
        val thirdLayerSize = source.nextInt(10, 100)

        val sampleCount = source.nextInt(2, 50)
        val epochsCount = source.nextInt(5, 50)

        val input = FloatMatrix(inputSize, sampleCount)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, sampleCount)
        expected.fillRandom(source)

        val firstLayer =
            DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
        val secondLayer =
            DenseLayer(
                secondLayerSize,
                thirdLayerSize,
                LeakyLeRU(secondLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )
        val thirdLayer =
            DenseLayer(
                thirdLayerSize,
                outputSize,
                LeakyLeRU(thirdLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )

        val cores = source.nextInt(1, 10)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            firstLayer, secondLayer, thirdLayer
        )

        var firstLayerWeights = FloatMatrix(secondLayerSize, inputSize, firstLayer.weights)
        var firstLayerBiases = FloatVector(firstLayer.biases)

        var secondLayerWeights = FloatMatrix(thirdLayerSize, secondLayerSize, secondLayer.weights)
        var secondLayerBiases = FloatVector(secondLayer.biases)

        var thirdLayerWeights = FloatMatrix(outputSize, thirdLayerSize, thirdLayer.weights)
        var thirdLayerBiases = FloatVector(thirdLayer.biases)

        for (epoch in 0 until epochsCount) {

            val firstZ = firstLayerWeights * input + firstLayerBiases.broadcast(sampleCount)
            val firstPrediction = leakyLeRU(firstZ, leRUGradient)

            val secondZ = secondLayerWeights * firstPrediction + secondLayerBiases.broadcast(sampleCount)
            val secondPrediction = leakyLeRU(secondZ, leRUGradient)

            val thirdZ = thirdLayerWeights * secondPrediction + thirdLayerBiases.broadcast(sampleCount)
            val thirdPrediction = leakyLeRU(thirdZ, leRUGradient)

            val thirdLayerCostError = mseCostFunctionDerivative(thirdPrediction, expected)
            val thirdLayerError = thirdLayerCostError.hadamardMul(leakyLeRUDerivative(thirdZ, leRUGradient))

            val secondLayerError = (thirdLayerWeights.transpose() * thirdLayerError).hadamardMul(
                leakyLeRUDerivative(
                    secondZ,
                    leRUGradient
                )
            )

            val firstLayerError = (secondLayerWeights.transpose() * secondLayerError).hadamardMul(
                leakyLeRUDerivative(
                    firstZ,
                    leRUGradient
                )
            )

            val firstLayerWeightsDelta = firstLayerError * input.transpose()
            val firstLayerBiasesDelta = firstLayerError

            firstLayerWeights -= firstLayerWeightsDelta * alpha / sampleCount
            firstLayerBiases -= firstLayerBiasesDelta.reduce() * alpha / sampleCount

            val secondLayerWeightsDelta = secondLayerError * firstPrediction.transpose()
            val secondLayerBiasesDelta = secondLayerError

            secondLayerWeights -= secondLayerWeightsDelta * alpha / sampleCount
            secondLayerBiases -= secondLayerBiasesDelta.reduce() * alpha / sampleCount

            val thirdLayerWeightsDelta = thirdLayerError * secondPrediction.transpose()
            val thirdLayerBiasesDelta = thirdLayerError

            thirdLayerWeights -= thirdLayerWeightsDelta * alpha / sampleCount
            thirdLayerBiases -= thirdLayerBiasesDelta.reduce() * alpha / sampleCount
        }

        neuralNetwork.fit(
            input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize,
            sampleCount, sampleCount,
            epochsCount, alpha, -1, false, Float.MIN_VALUE
        )

        Assertions.assertArrayEquals(firstLayerWeights.toFlatArray(), firstLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(firstLayerBiases.toArray(), firstLayer.biases, 0.0001f)

        Assertions.assertArrayEquals(secondLayerWeights.toFlatArray(), secondLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(secondLayerBiases.toArray(), secondLayer.biases, 0.0001f)

        Assertions.assertArrayEquals(thirdLayerWeights.toFlatArray(), thirdLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(thirdLayerBiases.toArray(), thirdLayer.biases, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun threeLayersMultiSampleTestSeveralEpochsMiniBatch(
        seed: Long,
        firstLayerSeed: Long,
        secondLayerSeed: Long,
        thirdLayerSeed: Long
    ) {
        val alpha = 0.0005f
        val leRUGradient = 0.01f

        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val secondLayerSize = source.nextInt(10, 100)
        val thirdLayerSize = source.nextInt(10, 100)

        val samplesCount = source.nextInt(10, 100)
        val miniBatchSize = samplesCount / 10
        val epochsCount = source.nextInt(5, 50)

        val input = FloatMatrix(inputSize, samplesCount)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, samplesCount)
        expected.fillRandom(source)

        val firstLayer =
            DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
        val secondLayer =
            DenseLayer(
                secondLayerSize,
                thirdLayerSize,
                LeakyLeRU(secondLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )
        val thirdLayer =
            DenseLayer(
                thirdLayerSize,
                outputSize,
                LeakyLeRU(thirdLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )

        val cores = source.nextInt(1, 10)
        val neuralNetwork = NeuralNetwork(
            MSECostFunction(), cores,
            firstLayer, secondLayer, thirdLayer
        )

        var firstLayerWeights = FloatMatrix(secondLayerSize, inputSize, firstLayer.weights)
        var firstLayerBiases = FloatVector(firstLayer.biases)

        var secondLayerWeights = FloatMatrix(thirdLayerSize, secondLayerSize, secondLayer.weights)
        var secondLayerBiases = FloatVector(secondLayer.biases)

        var thirdLayerWeights = FloatMatrix(outputSize, thirdLayerSize, thirdLayer.weights)
        var thirdLayerBiases = FloatVector(thirdLayer.biases)

        for (epoch in 0 until epochsCount) {
            for (i in 0 until samplesCount step miniBatchSize) {
                val miniSampleSize = min(miniBatchSize, samplesCount - i)

                val firstZ =
                    firstLayerWeights * input.subMatrix(i, miniSampleSize) + firstLayerBiases.broadcast(miniSampleSize)
                val firstPrediction = leakyLeRU(firstZ, leRUGradient)

                val secondZ = secondLayerWeights * firstPrediction + secondLayerBiases.broadcast(miniSampleSize)
                val secondPrediction = leakyLeRU(secondZ, leRUGradient)

                val thirdZ = thirdLayerWeights * secondPrediction + thirdLayerBiases.broadcast(miniSampleSize)
                val thirdPrediction = leakyLeRU(thirdZ, leRUGradient)

                val thirdLayerCostError =
                    mseCostFunctionDerivative(thirdPrediction, expected.subMatrix(i, miniSampleSize))
                val thirdLayerError = thirdLayerCostError.hadamardMul(leakyLeRUDerivative(thirdZ, leRUGradient))

                val secondLayerError = (thirdLayerWeights.transpose() * thirdLayerError).hadamardMul(
                    leakyLeRUDerivative(
                        secondZ,
                        leRUGradient
                    )
                )

                val firstLayerError = (secondLayerWeights.transpose() * secondLayerError).hadamardMul(
                    leakyLeRUDerivative(
                        firstZ,
                        leRUGradient
                    )
                )

                val firstLayerWeightsDelta = firstLayerError * input.subMatrix(i, miniSampleSize).transpose()
                val firstLayerBiasesDelta = firstLayerError

                firstLayerWeights -= firstLayerWeightsDelta * alpha / miniSampleSize
                firstLayerBiases -= firstLayerBiasesDelta.reduce() * alpha / miniSampleSize

                val secondLayerWeightsDelta = secondLayerError * firstPrediction.transpose()
                val secondLayerBiasesDelta = secondLayerError

                secondLayerWeights -= secondLayerWeightsDelta * alpha / miniSampleSize
                secondLayerBiases -= secondLayerBiasesDelta.reduce() * alpha / miniSampleSize

                val thirdLayerWeightsDelta = thirdLayerError * secondPrediction.transpose()
                val thirdLayerBiasesDelta = thirdLayerError

                thirdLayerWeights -= thirdLayerWeightsDelta * alpha / miniSampleSize
                thirdLayerBiases -= thirdLayerBiasesDelta.reduce() * alpha / miniSampleSize
            }
        }

        neuralNetwork.fit(
            input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize,
            samplesCount, miniBatchSize,
            epochsCount, alpha, -1, false, Float.MIN_VALUE
        )

        Assertions.assertArrayEquals(firstLayerWeights.toFlatArray(), firstLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(firstLayerBiases.toArray(), firstLayer.biases, 0.0001f)

        Assertions.assertArrayEquals(secondLayerWeights.toFlatArray(), secondLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(secondLayerBiases.toArray(), secondLayer.biases, 0.0001f)

        Assertions.assertArrayEquals(thirdLayerWeights.toFlatArray(), thirdLayer.weights, 0.0001f)
        Assertions.assertArrayEquals(thirdLayerBiases.toArray(), thirdLayer.biases, 0.0001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun threeLayersConstFunctionTest(
        seed: Long,
        firstLayerSeed: Long,
        secondLayerSeed: Long,
        thirdLayerSeed: Long
    ) {
        val leRUGradient = 0.01f
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val secondLayerSize = source.nextInt(10, 100)
        val thirdLayerSize = source.nextInt(10, 100)

        val sampleCount = source.nextInt(2, 50)

        val input = FloatMatrix(inputSize, sampleCount)
        input.fillRandom(source)

        val expected = FloatMatrix(outputSize, sampleCount)
        expected.fillRandom(source)

        val firstLayer =
            DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
        val secondLayer =
            DenseLayer(
                secondLayerSize,
                thirdLayerSize,
                LeakyLeRU(secondLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )
        val thirdLayer =
            DenseLayer(
                thirdLayerSize,
                outputSize,
                LeakyLeRU(thirdLayerSeed),
                WeightsOptimizer.OptimizerType.SIMPLE
            )
        val firstLayerWeights = FloatMatrix(secondLayerSize, inputSize, firstLayer.weights)
        val firstLayerBiases = FloatVector(firstLayer.biases)

        val secondLayerWeights = FloatMatrix(thirdLayerSize, secondLayerSize, secondLayer.weights)
        val secondLayerBiases = FloatVector(secondLayer.biases)

        val thirdLayerWeights = FloatMatrix(outputSize, thirdLayerSize, thirdLayer.weights)
        val thirdLayerBiases = FloatVector(thirdLayer.biases)

        val firstZ = firstLayerWeights * input + firstLayerBiases.broadcast(sampleCount)
        val firstPrediction = leakyLeRU(firstZ, leRUGradient)

        val secondZ = secondLayerWeights * firstPrediction + secondLayerBiases.broadcast(sampleCount)
        val secondPrediction = leakyLeRU(secondZ, leRUGradient)

        val thirdZ = thirdLayerWeights * secondPrediction + thirdLayerBiases.broadcast(sampleCount)
        val thirdPrediction = leakyLeRU(thirdZ, leRUGradient)

        val expectedCost = mseCostFunction(thirdPrediction, expected)
        val cores = source.nextInt(1, 10)
        val cost = Executors.newFixedThreadPool(cores).use { executor ->
            NeuralNetwork.trainingCost(
                arrayOf(firstLayer, secondLayer, thirdLayer), MSECostFunction(),
                max(max(outputSize, secondLayerSize), thirdLayerSize), sampleCount, input.toFlatArray(),
                expected.toFlatArray(), executor, cores
            )
        }

        Assertions.assertEquals(expectedCost, cost, 0.0001f)
    }

}