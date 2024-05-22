package com.tornadoml.cpu

import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import java.nio.ByteBuffer
import java.security.SecureRandom

class NeuralNetworkTests {
    private val securesRandom = SecureRandom()

    @Test
    fun singleLayerSingleSampleMSETestOneEpoch() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("singleLayerSingleSampleTests seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val inputSize = source.nextInt(1, 100)
            val outputSize = source.nextInt(1, 100)

            val input = FloatMatrix(inputSize, 1)
            input.fillRandom(source)

            val expected = FloatMatrix(outputSize, 1)
            expected.fillRandom(source)

            val layer =
                DenseLayer(inputSize, outputSize, LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE)
            val neuralNetwork = NeuralNetwork(
                MSECostFunction(),
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

            val alpha = 0.01f

            weights -= weightsDelta * alpha
            biases -= biasesDelta * alpha

            neuralNetwork.train(
                input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize, 1, 1,
                1, 0.01f, -1, false
            )

            Assertions.assertArrayEquals(weights.toFlatArray(), layer.weights, 0.0001f)
            Assertions.assertArrayEquals(biases.toFlatArray(), layer.biases, 0.0001f)
        }
    }

    @Test
    fun singleLayerMultiSampleMSETestOneEpoch() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("singleLayerMultiSampleMSETestOneEpoch seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val sampleSize = source.nextInt(2, 50)

            val inputSize = source.nextInt(sampleSize, 100)
            val outputSize = source.nextInt(sampleSize, 100)

            val input = FloatMatrix(inputSize, sampleSize)
            input.fillRandom(source)

            val expected = FloatMatrix(outputSize, sampleSize)
            expected.fillRandom(source)

            val layer =
                DenseLayer(inputSize, outputSize, LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE)
            val neuralNetwork = NeuralNetwork(
                MSECostFunction(),
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

            val alpha = 0.01f

            weights -= weightsDelta * alpha / sampleSize
            biases -= biasesDelta.reduce() / sampleSize * alpha

            neuralNetwork.train(
                input.transpose().toArray(),
                expected.transpose().toArray(),
                inputSize,
                outputSize,
                sampleSize,
                sampleSize,
                1,
                alpha,
                -1,
                false
            )

            Assertions.assertArrayEquals(weights.toFlatArray(), layer.weights, 0.0001f)
            Assertions.assertArrayEquals(biases.toArray(), layer.biases, 0.0001f)
        }
    }

    @Test
    fun singleLayerMultiSampleMSETestSeveralEpochs() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("singleLayerMultiSampleMSETestSeveralEpochs seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val sampleSize = source.nextInt(2, 50)
            val epochs = source.nextInt(5, 50)

            val inputSize = source.nextInt(sampleSize, 100)
            val outputSize = source.nextInt(sampleSize, 100)

            val input = FloatMatrix(inputSize, sampleSize)
            input.fillRandom(source)

            val expected = FloatMatrix(outputSize, sampleSize)
            expected.fillRandom(source)

            val layer =
                DenseLayer(inputSize, outputSize, LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE)
            val neuralNetwork = NeuralNetwork(
                MSECostFunction(),
                layer
            )

            var weights = FloatMatrix(outputSize, inputSize, layer.weights)
            var biases = FloatVector(layer.biases)
            val alpha = 0.01f

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

            neuralNetwork.train(
                input.transpose().toArray(),
                expected.transpose().toArray(),
                inputSize,
                outputSize,
                sampleSize,
                sampleSize,
                epochs,
                alpha,
                -1,
                false
            )

            Assertions.assertArrayEquals(weights.toFlatArray(), layer.weights, 0.0001f)
            Assertions.assertArrayEquals(biases.toArray(), layer.biases, 0.0001f)
        }
    }

    @Test
    fun twoLayersSingleSampleMSETestOneEpoch() {
        for (n in 0..9) {
            var bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            val alpha = 0.01f

            println("singleLayerSingleSampleTests seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val inputSize = source.nextInt(1, 100)
            val outputSize = source.nextInt(1, 100)

            val secondLayerSize = source.nextInt(10, 100)

            val input = FloatMatrix(inputSize, 1)
            input.fillRandom(source)

            val expected = FloatMatrix(outputSize, 1)
            expected.fillRandom(source)

            bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val firstLayerSeed = bBuffer.getLong()
            bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val secondLayerSeed = bBuffer.getLong()

            println("firstLayerSeed: $firstLayerSeed, secondLayerSeed: $secondLayerSeed")

            val firstLayer =
                DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
            val secondLayer =
                DenseLayer(
                    secondLayerSize,
                    outputSize,
                    LeakyLeRU(secondLayerSeed),
                    WeightsOptimizer.OptimizerType.SIMPLE
                )

            val neuralNetwork = NeuralNetwork(
                MSECostFunction(),
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


            neuralNetwork.train(
                input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize, 1, 1,
                1, alpha, -1, false
            )

            Assertions.assertArrayEquals(firstLayerWeights.toFlatArray(), firstLayer.weights, 0.0001f)
            Assertions.assertArrayEquals(firstLayerBiases.toFlatArray(), firstLayer.biases, 0.0001f)

            Assertions.assertArrayEquals(secondLayerWeights.toFlatArray(), secondLayer.weights, 0.0001f)
            Assertions.assertArrayEquals(secondLayerBiases.toFlatArray(), secondLayer.biases, 0.0001f)
        }
    }

    @Test
    fun twoLayersSeveralSamplesMSETestOneEpoch() {
        for (n in 0..9) {
            var bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            val alpha = 0.01f

            println("twoLayersSeveralSamplesMSETestOneEpoch seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val inputSize = source.nextInt(1, 100)
            val outputSize = source.nextInt(1, 100)
            val samplesCount = source.nextInt(2, 50)

            val secondLayerSize = source.nextInt(10, 100)

            val input = FloatMatrix(inputSize, samplesCount)
            input.fillRandom(source)

            val expected = FloatMatrix(outputSize, samplesCount)
            expected.fillRandom(source)

            bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val firstLayerSeed = bBuffer.getLong()
            bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val secondLayerSeed = bBuffer.getLong()

            println("firstLayerSeed: $firstLayerSeed, secondLayerSeed: $secondLayerSeed")

            val firstLayer =
                DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
            val secondLayer =
                DenseLayer(
                    secondLayerSize,
                    outputSize,
                    LeakyLeRU(secondLayerSeed),
                    WeightsOptimizer.OptimizerType.SIMPLE
                )

            val neuralNetwork = NeuralNetwork(
                MSECostFunction(),
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


            neuralNetwork.train(
                input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize, samplesCount,
                samplesCount, 1, alpha, -1, false
            )

            Assertions.assertArrayEquals(firstLayerWeights.toFlatArray(), firstLayer.weights, 0.0001f)
            Assertions.assertArrayEquals(firstLayerBiases.toArray(), firstLayer.biases, 0.0001f)

            Assertions.assertArrayEquals(secondLayerWeights.toFlatArray(), secondLayer.weights, 0.0001f)
            Assertions.assertArrayEquals(secondLayerBiases.toArray(), secondLayer.biases, 0.0001f)
        }
    }

    @Test
    fun twoLayersSeveralSamplesMSETestSeveralEpochs() {
        for (n in 0..9) {
            var bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            val alpha = 0.01f

            println("twoLayersSeveralSamplesMSETestOneEpoch seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val inputSize = source.nextInt(1, 100)
            val outputSize = source.nextInt(1, 100)
            val samplesCount = source.nextInt(2, 50)

            val secondLayerSize = source.nextInt(10, 100)

            val input = FloatMatrix(inputSize, samplesCount)
            input.fillRandom(source)

            val expected = FloatMatrix(outputSize, samplesCount)
            expected.fillRandom(source)

            bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val firstLayerSeed = bBuffer.getLong()
            bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val secondLayerSeed = bBuffer.getLong()

            println("firstLayerSeed: $firstLayerSeed, secondLayerSeed: $secondLayerSeed")

            val firstLayer =
                DenseLayer(inputSize, secondLayerSize, LeakyLeRU(firstLayerSeed), WeightsOptimizer.OptimizerType.SIMPLE)
            val secondLayer =
                DenseLayer(
                    secondLayerSize,
                    outputSize,
                    LeakyLeRU(secondLayerSeed),
                    WeightsOptimizer.OptimizerType.SIMPLE
                )

            val neuralNetwork = NeuralNetwork(
                MSECostFunction(),
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

            neuralNetwork.train(
                input.transpose().toArray(), expected.transpose().toArray(), inputSize, outputSize, samplesCount,
                samplesCount, epochs, alpha, -1, false
            )

            Assertions.assertArrayEquals(firstLayerWeights.toFlatArray(), firstLayer.weights, 0.0001f)
            Assertions.assertArrayEquals(firstLayerBiases.toArray(), firstLayer.biases, 0.0001f)

            Assertions.assertArrayEquals(secondLayerWeights.toFlatArray(), secondLayer.weights, 0.0001f)
            Assertions.assertArrayEquals(secondLayerBiases.toArray(), secondLayer.biases, 0.0001f)
        }
    }
}