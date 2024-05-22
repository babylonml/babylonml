package com.tornadoml.cpu

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import com.tornadoml.mnist.MNISTLoader
import org.apache.commons.rng.simple.RandomSource
import java.nio.ByteBuffer
import java.security.SecureRandom
import java.util.*
import kotlin.math.max

class DenseLayerTests {
    private val securesRandom = SecureRandom()

    @Test
    fun predictSingleSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("predictTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val input = FloatMatrix(layer.inputSize, 1)
            input.fillRandom(source)

            val weights = FloatMatrix(layer.outputSize, layer.inputSize, layer.weights)
            val biases = FloatMatrix(layer.outputSize, 1, layer.biases)

            val z = weights * input + biases
            Assertions.assertEquals(z.rows, layer.outputSize)
            Assertions.assertEquals(z.cols, 1)

            val a = leakyLeRU(z, 0.01f)

            val result = FloatArray(layer.outputSize) {
                source.nextFloat()
            }

            layer.predict(input.toFlatArray(), result, 1)

            Assertions.assertArrayEquals(a.toFlatArray(), result, 0.001f)
        }
    }

    @Test
    fun predictMultiSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("predictMultiSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val sampleCount = source.nextInt(2, 10)
            val input = FloatMatrix(layer.inputSize, sampleCount)
            input.fillRandom(source)

            val weights = FloatMatrix(layer.outputSize, layer.inputSize, layer.weights)
            val biases = FloatVector(layer.biases)

            val z = weights * input + biases.broadcast(sampleCount)
            Assertions.assertEquals(z.rows, layer.outputSize)
            Assertions.assertEquals(z.cols, sampleCount)

            val a = leakyLeRU(z, 0.01f)

            val result = FloatArray(layer.outputSize * sampleCount) {
                source.nextFloat()
            }

            layer.predict(input.toFlatArray(), result, sampleCount)

            Assertions.assertArrayEquals(a.toFlatArray(), result, 0.001f)
        }
    }

    @Test
    fun backwardLastLayerSingleSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("backwardLastLayerSingleSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val weights = FloatMatrix(layer.outputSize, layer.inputSize, layer.weights)
            val biases = FloatVector(layer.biases)

            val dEdY = FloatMatrix(layer.outputSize, 1)
            dEdY.fillRandom(source)

            val x = FloatMatrix(layer.inputSize, 1)
            x.fillRandom(source)

            val z = weights * x + biases.broadcast(1)
            val dZ = leakyLeRUDerivative(z, 0.01f)
            val dEdZ = dEdY.hadamardMul(dZ)

            val dEdW = dEdZ * x.transpose()
            val dEdB = dEdZ

            val prevZ = FloatMatrix(layer.inputSize, 1)
            val prevDeDz = (weights.transpose() * dEdZ).hadamardMul(leakyLeRUDerivative(prevZ, 0.01f))

            val calculatedWeightDelta = FloatArray(layer.inputSize * layer.outputSize) {
                source.nextFloat()
            }
            val calculatedBiasesDelta = FloatArray(layer.outputSize) {
                source.nextFloat()
            }

            val costError = FloatArray(max(layer.outputSize, layer.inputSize)) {
                source.nextFloat()
            }

            layer.backwardLastLayer(
                x.toFlatArray(), prevZ.toFlatArray(), z.toFlatArray(),
                dEdY.toFlatArray().copyInto(costError), costError, calculatedWeightDelta, calculatedBiasesDelta, 1
            )

            Assertions.assertArrayEquals(dEdW.toFlatArray(), calculatedWeightDelta, 0.001f)
            Assertions.assertArrayEquals(dEdB.toFlatArray(), calculatedBiasesDelta, 0.001f)
            Assertions.assertArrayEquals(prevDeDz.toFlatArray(), costError.copyOf(layer.inputSize), 0.001f)
        }
    }

    @Test
    fun backwardLastLayerMultiSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("backwardLastLayerMultiSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val weights = FloatMatrix(layer.outputSize, layer.inputSize, layer.weights)
            val biases = FloatVector(layer.biases)

            val sampleSize = source.nextInt(2, 10)

            val dEdY = FloatMatrix(layer.outputSize, sampleSize)
            dEdY.fillRandom(source)

            val x = FloatMatrix(layer.inputSize, sampleSize)
            x.fillRandom(source)

            val z = weights * x + biases.broadcast(sampleSize)
            val dZ = leakyLeRUDerivative(z, 0.01f)
            val dEdZ = dEdY.hadamardMul(dZ)

            val dEdW = dEdZ * x.transpose()
            val dEdB = dEdZ

            val prevZ = FloatMatrix(layer.inputSize, sampleSize)
            val prevDeDz = (weights.transpose() * dEdZ).hadamardMul(leakyLeRUDerivative(prevZ, 0.01f))

            val calculatedWeightDelta = FloatArray(layer.inputSize * layer.outputSize) {
                source.nextFloat()
            }
            val calculatedBiasesDelta = FloatArray(layer.outputSize * sampleSize) {
                source.nextFloat()
            }

            val costError = FloatArray(max(layer.outputSize * sampleSize, layer.inputSize * sampleSize)) {
                source.nextFloat()
            }

            layer.backwardLastLayer(
                x.toFlatArray(), prevZ.toFlatArray(), z.toFlatArray(),
                dEdY.toFlatArray().copyInto(costError), costError, calculatedWeightDelta, calculatedBiasesDelta,
                sampleSize
            )

            Assertions.assertArrayEquals(dEdW.toFlatArray(), calculatedWeightDelta, 0.001f)
            Assertions.assertArrayEquals(dEdB.toFlatArray(), calculatedBiasesDelta, 0.001f)
            Assertions.assertArrayEquals(prevDeDz.toFlatArray(), costError.copyOf(layer.inputSize * sampleSize), 0.001f)
        }
    }

    @Test
    fun forwardTrainingSingleSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("forwardTrainingSingleSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val input = FloatMatrix(layer.inputSize, 1)
            input.fillRandom(source)

            val weights = FloatMatrix(layer.outputSize, layer.inputSize, layer.weights)
            val biases = FloatMatrix(layer.outputSize, 1, layer.biases)

            val z = weights * input + biases
            Assertions.assertEquals(z.rows, layer.outputSize)
            Assertions.assertEquals(z.cols, 1)

            val a = leakyLeRU(z, 0.01f)

            val activationArgument = FloatArray(layer.outputSize)
            val prediction = FloatArray(layer.outputSize)

            val inputArray = FloatArray(layer.inputSize + 1) {
                source.nextFloat()
            }

            layer.forwardTraining(
                input.toFlatArray().copyInto(inputArray, 1),
                1, activationArgument, prediction, 1
            )

            Assertions.assertArrayEquals(z.toFlatArray(), activationArgument, 0.001f)
            Assertions.assertArrayEquals(a.toFlatArray(), prediction, 0.001f)
        }
    }

    @Test
    fun forwardTrainingMultiSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("forwardTrainingMultiSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val sampleCount = source.nextInt(2, 10)
            val input = FloatMatrix(layer.inputSize, sampleCount)
            input.fillRandom(source)

            val weights = FloatMatrix(layer.outputSize, layer.inputSize, layer.weights)
            val biases = FloatVector(layer.biases)

            val z = weights * input + biases.broadcast(sampleCount)
            Assertions.assertEquals(z.rows, layer.outputSize)
            Assertions.assertEquals(z.cols, sampleCount)

            val a = leakyLeRU(z, 0.01f)

            val activationArgument = FloatArray(layer.outputSize * sampleCount)
            val prediction = FloatArray(layer.outputSize * sampleCount)

            val inputArray = FloatArray(layer.inputSize * sampleCount + 1) {
                source.nextFloat()
            }

            layer.forwardTraining(
                input.toFlatArray().copyInto(inputArray, 1),
                1, activationArgument, prediction, sampleCount
            )

            Assertions.assertArrayEquals(z.toFlatArray(), activationArgument, 0.001f)
            Assertions.assertArrayEquals(a.toFlatArray(), prediction, 0.001f)
        }
    }

    @Test
    fun backwardLastLayerNoErrorSingleSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("backwardLastLayerNoErrorSingleSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val weights = FloatMatrix(layer.outputSize, layer.inputSize, layer.weights)
            val biases = FloatVector(layer.biases)

            val dEdY = FloatMatrix(layer.outputSize, 1)
            dEdY.fillRandom(source)

            val x = FloatMatrix(layer.inputSize, 1)
            x.fillRandom(source)

            val z = weights * x + biases.broadcast(1)
            val dZ = leakyLeRUDerivative(z, 0.01f)
            val dEdZ = dEdY.hadamardMul(dZ)

            val dEdW = dEdZ * x.transpose()
            val dEdB = dEdZ


            val calculatedWeightDelta = FloatArray(layer.inputSize * layer.outputSize) {
                source.nextFloat()
            }
            val calculatedBiasesDelta = FloatArray(layer.outputSize) {
                source.nextFloat()
            }

            val costError = FloatArray(max(layer.outputSize, layer.inputSize)) {
                source.nextFloat()
            }

            layer.backwardLastLayerNoError(
                x.toFlatArray(), z.toFlatArray(),
                dEdY.toFlatArray().copyInto(costError), calculatedWeightDelta, calculatedBiasesDelta, 1
            )

            Assertions.assertArrayEquals(dEdW.toFlatArray(), calculatedWeightDelta, 0.001f)
            Assertions.assertArrayEquals(dEdB.toFlatArray(), calculatedBiasesDelta, 0.001f)
        }
    }

    @Test
    fun backwardLastLayerNoErrorMultiSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("backwardLastLayerNoErrorMultiSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val weights = FloatMatrix(layer.outputSize, layer.inputSize, layer.weights)
            val biases = FloatVector(layer.biases)

            val sampleSize = source.nextInt(2, 10)
            val dEdY = FloatMatrix(layer.outputSize, sampleSize)
            dEdY.fillRandom(source)

            val x = FloatMatrix(layer.inputSize, sampleSize)
            x.fillRandom(source)

            val z = weights * x + biases.broadcast(sampleSize)
            val dZ = leakyLeRUDerivative(z, 0.01f)
            val dEdZ = dEdY.hadamardMul(dZ)

            val dEdW = dEdZ * x.transpose()
            val dEdB = dEdZ


            val calculatedWeightDelta = FloatArray(layer.inputSize * layer.outputSize) {
                source.nextFloat()
            }
            val calculatedBiasesDelta = FloatArray(layer.outputSize * sampleSize) {
                source.nextFloat()
            }

            val costError = FloatArray(max(layer.outputSize * sampleSize, layer.inputSize * sampleSize)) {
                source.nextFloat()
            }

            layer.backwardLastLayerNoError(
                x.toFlatArray(), z.toFlatArray(),
                dEdY.toFlatArray().copyInto(costError), calculatedWeightDelta, calculatedBiasesDelta, sampleSize
            )

            Assertions.assertArrayEquals(dEdW.toFlatArray(), calculatedWeightDelta, 0.001f)
            Assertions.assertArrayEquals(dEdB.toFlatArray(), calculatedBiasesDelta, 0.001f)
        }
    }

    @Test
    fun backwardMiddleLayerSingleSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("backwardMiddleLayerSingleSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val weights = FloatMatrix(layer.outputSize, layer.inputSize, layer.weights)

            val x = FloatMatrix(layer.inputSize, 1)
            x.fillRandom(source)

            val prevZ = FloatMatrix(layer.inputSize, 1)
            prevZ.fillRandom(source)

            val dLdZ = FloatMatrix(layer.outputSize, 1)
            dLdZ.fillRandom(source)

            val dLdW = dLdZ * x.transpose()
            val dLdB = dLdZ

            val prevDlDz = (weights.transpose() * dLdZ).hadamardMul(leakyLeRUDerivative(prevZ, 0.01f))

            val weightsDelta = FloatArray(layer.inputSize * layer.outputSize) {
                source.nextFloat()
            }
            val biasesDelta = FloatArray(layer.outputSize) {
                source.nextFloat()
            }

            val costErrors = FloatArray(max(layer.outputSize, layer.inputSize)) {
                source.nextFloat()
            }

            layer.backwardMiddleLayer(
                x.toFlatArray(), dLdZ.toFlatArray(),
                prevZ.toFlatArray(), costErrors,
                weightsDelta, biasesDelta, 1
            )

            Assertions.assertArrayEquals(dLdW.toFlatArray(), weightsDelta, 0.001f)
            Assertions.assertArrayEquals(dLdB.toFlatArray(), biasesDelta, 0.001f)
            Assertions.assertArrayEquals(prevDlDz.toFlatArray(), costErrors.copyOf(layer.inputSize), 0.001f)
        }
    }

    @Test
    fun backwardMiddleLayerMultiSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("backwardMiddleLayerMultiSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val weights = FloatMatrix(layer.outputSize, layer.inputSize, layer.weights)
            val sampleSize = source.nextInt(2, 10)

            val x = FloatMatrix(layer.inputSize, sampleSize)
            x.fillRandom(source)

            val prevZ = FloatMatrix(layer.inputSize, sampleSize)
            prevZ.fillRandom(source)

            val dLdZ = FloatMatrix(layer.outputSize, sampleSize)
            dLdZ.fillRandom(source)

            val dLdW = dLdZ * x.transpose()
            val dLdB = dLdZ

            val prevDlDz = (weights.transpose() * dLdZ).hadamardMul(leakyLeRUDerivative(prevZ, 0.01f))

            val weightsDelta = FloatArray(layer.inputSize * layer.outputSize) {
                source.nextFloat()
            }
            val biasesDelta = FloatArray(layer.outputSize * sampleSize) {
                source.nextFloat()
            }

            val costErrors = FloatArray(max(layer.outputSize * sampleSize, layer.inputSize * sampleSize)) {
                source.nextFloat()
            }

            layer.backwardMiddleLayer(
                x.toFlatArray(), dLdZ.toFlatArray(),
                prevZ.toFlatArray(), costErrors,
                weightsDelta, biasesDelta, sampleSize
            )

            Assertions.assertArrayEquals(dLdW.toFlatArray(), weightsDelta, 0.001f)
            Assertions.assertArrayEquals(dLdB.toFlatArray(), biasesDelta, 0.001f)
            Assertions.assertArrayEquals(
                prevDlDz.toFlatArray(),
                costErrors.copyOf(layer.inputSize * sampleSize),
                0.001f
            )
        }
    }

    @Test
    fun backwardZeroLayerSingleSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("backwardZeroLayerSingleSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val x = FloatMatrix(layer.inputSize, 1)
            x.fillRandom(source)

            val dLdZ = FloatMatrix(layer.outputSize, 1)
            dLdZ.fillRandom(source)

            val dLdW = dLdZ * x.transpose()
            val dLdB = dLdZ

            val weightsDelta = FloatArray(layer.inputSize * layer.outputSize) {
                source.nextFloat()
            }
            val biasesDelta = FloatArray(layer.outputSize) {
                source.nextFloat()
            }

            val inputArray = FloatArray(layer.inputSize + 2) {
                source.nextFloat()
            }

            layer.backwardZeroLayer(
                x.toFlatArray().copyInto(inputArray, 1), 1, dLdZ.toFlatArray(),
                weightsDelta, biasesDelta, 1
            )

            Assertions.assertArrayEquals(dLdW.toFlatArray(), weightsDelta, 0.001f)
            Assertions.assertArrayEquals(dLdB.toFlatArray(), biasesDelta, 0.001f)
        }
    }

    @Test
    fun backwardZeroLayerMultiSampleSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("backwardZeroLayerMultiSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )
            val sampleSize = source.nextInt(2, 10)

            val x = FloatMatrix(layer.inputSize, sampleSize)
            x.fillRandom(source)

            val dLdZ = FloatMatrix(layer.outputSize, sampleSize)
            dLdZ.fillRandom(source)

            val dLdW = dLdZ * x.transpose()
            val dLdB = dLdZ

            val weightsDelta = FloatArray(layer.inputSize * layer.outputSize) {
                source.nextFloat()
            }
            val biasesDelta = FloatArray(layer.outputSize * sampleSize) {
                source.nextFloat()
            }

            val inputArray = FloatArray(layer.inputSize * sampleSize + 2) {
                source.nextFloat()
            }

            layer.backwardZeroLayer(
                x.toFlatArray().copyInto(inputArray, 1), 1, dLdZ.toFlatArray(),
                weightsDelta, biasesDelta, sampleSize
            )

            Assertions.assertArrayEquals(dLdW.toFlatArray(), weightsDelta, 0.001f)
            Assertions.assertArrayEquals(dLdB.toFlatArray(), biasesDelta, 0.001f)
        }
    }

    @Test
    fun updateWeightsAndBiasesSingleSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("backwardZeroLayerSingleSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val biasesDelta = FloatMatrix(layer.outputSize, 1)
            biasesDelta.fillRandom(source)

            val weightsDelta = FloatMatrix(layer.outputSize, layer.inputSize)
            weightsDelta.fillRandom(source)

            val weights = FloatMatrix(layer.outputSize, layer.inputSize, layer.weights)
            val biases = FloatMatrix(layer.outputSize, 1, layer.biases)

            val learningRate = source.nextFloat()

            val updatedWeights = weights - weightsDelta * learningRate
            val updatedBiases = biases - biasesDelta * learningRate

            layer.updateWeightsAndBiases(
                weightsDelta.toFlatArray(), biasesDelta.toFlatArray(),
                learningRate, 1
            )

            Assertions.assertArrayEquals(updatedWeights.toFlatArray(), layer.weights, 0.001f)
            Assertions.assertArrayEquals(updatedBiases.toFlatArray(), layer.biases, 0.001f)
        }
    }

    @Test
    fun updateWeightsAndBiasesMultiSampleTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("backwardZeroLayerSingleSampleTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val layer = DenseLayer(
                source.nextInt(1, 100), source.nextInt(1, 100),
                LeakyLeRU(source.nextLong()), WeightsOptimizer.OptimizerType.SIMPLE
            )

            val sampleSize = source.nextInt(2, 10)
            val biasesDelta = FloatMatrix(layer.outputSize, sampleSize)
            biasesDelta.fillRandom(source)

            val weightsDelta = FloatMatrix(layer.outputSize, layer.inputSize)
            weightsDelta.fillRandom(source)

            val weights = FloatMatrix(layer.outputSize, layer.inputSize, layer.weights)
            val biases = FloatVector(layer.biases)

            val learningRate = source.nextFloat()

            val updatedWeights = weights - weightsDelta * (learningRate / sampleSize)
            val updatedBiases = biases - (biasesDelta * (learningRate / sampleSize)).reduce()

            layer.updateWeightsAndBiases(
                weightsDelta.toFlatArray(), biasesDelta.toFlatArray(),
                learningRate, sampleSize
            )

            Assertions.assertArrayEquals(updatedWeights.toFlatArray(), layer.weights, 0.001f)
            Assertions.assertArrayEquals(updatedBiases.toArray(), layer.biases, 0.001f)
        }
    }


    @Test
    fun test10NeuronsSingleLayerRandomData() {
        val inputSize = 500
        val outputSize = 10
        val sampleSize = 100

        val seed = System.nanoTime()
        println("test10NeuronsSingleLayerRandomData seed: $seed")

        val random = Random(seed)

        val input = FloatArray(sampleSize * inputSize)
        val target = FloatArray(sampleSize * outputSize)

        for (i in 0 until sampleSize) {
            for (j in 0 until inputSize) {
                input[i * inputSize + j] = random.nextFloat()
            }

            target[i * outputSize + random.nextInt(outputSize)] = 1.0f
        }

        singleLayerTest(inputSize, outputSize, sampleSize, input, target, random)
    }

    @Test
    @Throws(Exception::class)
    fun test10NeuronsSingleLayerMNISTData() {
        val inputSize = 784
        val outputSize = 10
        val sampleSize = 100

        val seed = System.nanoTime()
        println("test10NeuronsSingleLayerMNISTData seed: $seed")

        val random = Random(seed)


        val input = FloatArray(sampleSize * inputSize)
        val target = FloatArray(sampleSize * outputSize)

        val mnistImages = MNISTLoader.loadMNISTImages()
        val mnistLabels = MNISTLoader.loadMNISTLabels()

        for (i in 0 until sampleSize) {
            System.arraycopy(mnistImages[i], 0, input, i * inputSize, inputSize)
            target[i * outputSize + mnistLabels[i]] = 1.0f
        }

        singleLayerTest(inputSize, outputSize, sampleSize, input, target, random)
    }

    @Test
    fun test60on10NeuronsTwoLayerRandomData() {
        val inputSize = 500
        val outputSize = 10
        val sampleSize = 100
        val hiddenLayerSize = 60

        val seed = System.nanoTime()
        println("test60on10NeuronsTwoLayerRandomData seed: $seed")

        val random = Random(seed)

        val input = FloatArray(sampleSize * inputSize)
        val target = FloatArray(sampleSize * outputSize)

        for (i in 0 until sampleSize) {
            for (j in 0 until inputSize) {
                input[i * inputSize + j] = random.nextFloat()
            }

            target[i * outputSize + random.nextInt(outputSize)] = 1.0f
        }

        twoLayersTest(inputSize, hiddenLayerSize, outputSize, sampleSize, input, target, random)
    }

    @Test
    @Throws(Exception::class)
    fun test60on10NeuronsTwoLayerMNISTData() {
        val inputSize = 784
        val outputSize = 10
        val sampleSize = 100

        val hiddenLayerSize = 60

        val seed = System.nanoTime()
        println("test60on10NeuronsTwoLayerMNISTData seed: $seed")

        val random = Random(seed)


        val input = FloatArray(sampleSize * inputSize)
        val target = FloatArray(sampleSize * outputSize)

        val mnistImages = MNISTLoader.loadMNISTImages()
        val mnistLabels = MNISTLoader.loadMNISTLabels()

        for (i in 0 until sampleSize) {
            System.arraycopy(mnistImages[i], 0, input, i * inputSize, inputSize)
            target[i * outputSize + mnistLabels[i]] = 1.0f
        }

        twoLayersTest(inputSize, hiddenLayerSize, outputSize, sampleSize, input, target, random)
    }

    @Test
    fun test60on30on10NeuronsThreeLayerRandomData() {
        val inputSize = 500
        val outputSize = 10
        val sampleSize = 100

        val firstHiddenLayerSize = 60
        val secondHiddenLayerSize = 30

        val seed = System.nanoTime()
        println("test60on30on10NeuronsThreeLayerRandomData seed: $seed")

        val random = Random(seed)

        val input = FloatArray(sampleSize * inputSize)
        val target = FloatArray(sampleSize * outputSize)

        for (i in 0 until sampleSize) {
            for (j in 0 until inputSize) {
                input[i * inputSize + j] = random.nextFloat()
            }

            target[i * outputSize + random.nextInt(outputSize)] = 1.0f
        }

        nLayersTest(
            inputSize, intArrayOf(firstHiddenLayerSize, secondHiddenLayerSize),
            outputSize, sampleSize, input, target, random
        )
    }

    @Test
    @Throws(Exception::class)
    fun test60on30on10NeuronsThreeLayerMNISTData() {
        val inputSize = 784
        val outputSize = 10
        val sampleSize = 100

        val firstHiddenLayerSize = 60
        val secondHiddenLayerSize = 30

        val seed = System.nanoTime()
        println("test60on30on10NeuronsThreeLayerMNISTData seed: $seed")

        val random = Random(seed)


        val input = FloatArray(sampleSize * inputSize)
        val target = FloatArray(sampleSize * outputSize)

        val mnistImages = MNISTLoader.loadMNISTImages()
        val mnistLabels = MNISTLoader.loadMNISTLabels()

        for (i in 0 until sampleSize) {
            System.arraycopy(mnistImages[i], 0, input, i * inputSize, inputSize)
            target[i * outputSize + mnistLabels[i]] = 1.0f
        }

        nLayersTest(
            inputSize, intArrayOf(firstHiddenLayerSize, secondHiddenLayerSize),
            outputSize, sampleSize, input, target, random
        )
    }

    @Suppress("SameParameterValue")
    private fun singleLayerTest(
        inputSize: Int, outputSize: Int, sampleSize: Int, input: FloatArray,
        target: FloatArray, random: Random
    ) {
        val alpha = 0.01f
        val layer = DenseLayer(
            inputSize, outputSize, LeakyLeRU(random.nextLong()),
            WeightsOptimizer.OptimizerType.SIMPLE
        )
        val activations = FloatArray(sampleSize * outputSize)
        val predictions = FloatArray(sampleSize * outputSize)
        val weightsDelta = FloatArray(inputSize * outputSize)
        val biasesDelta = FloatArray(outputSize * sampleSize)
        val costFunction = MSECostFunction()
        val costs = FloatArray(outputSize * sampleSize)

        var cost = 0f
        for (i in 0..1000) {
            layer.forwardTraining(input, 0, activations, predictions, sampleSize)
            cost = costFunction.value(
                predictions, 0, target, 0, sampleSize * outputSize,
                sampleSize
            )
            if (i == 0) {
                println("Initial cost: $cost")
            }

            costFunction.derivative(
                predictions, 0, target, 0, costs, 0,
                sampleSize * outputSize
            )
            layer.backwardLastLayerNoError(
                input, activations, costs, weightsDelta, biasesDelta,
                sampleSize
            )

            layer.updateWeightsAndBiases(weightsDelta, biasesDelta, alpha, sampleSize)

            layer.forwardTraining(input, 0, activations, predictions, sampleSize)
            val newCost = costFunction.value(
                predictions, 0, target, 0, sampleSize * outputSize, sampleSize
            )
            Assertions.assertTrue(
                newCost < cost, "Cost increased: " + newCost + " > " +
                        cost + " on iteration " + i
            )
            cost = newCost
        }

        println("Cost: $cost")
    }

    @Suppress("SameParameterValue")
    private fun twoLayersTest(
        inputSize: Int, layerSize: Int, outputSize: Int, sampleSize: Int,
        input: FloatArray, target: FloatArray, random: Random
    ) {
        val alpha = 0.01f
        val firstLayer = DenseLayer(
            inputSize, layerSize, LeakyLeRU(random.nextLong()),
            WeightsOptimizer.OptimizerType.SIMPLE
        )
        val secondLayer = DenseLayer(
            layerSize, outputSize, LeakyLeRU(random.nextLong()),
            WeightsOptimizer.OptimizerType.SIMPLE
        )

        val maxInputSize = max(inputSize.toDouble(), layerSize.toDouble()).toInt()
        val maxOutputSize = max(layerSize.toDouble(), outputSize.toDouble()).toInt()

        val activations = arrayOfNulls<FloatArray>(2)
        val predictions = arrayOfNulls<FloatArray>(2)

        activations[0] = FloatArray(sampleSize * layerSize)
        predictions[0] = FloatArray(sampleSize * layerSize)

        activations[1] = FloatArray(sampleSize * outputSize)
        predictions[1] = FloatArray(sampleSize * outputSize)

        val weightsDelta = Array(2) { FloatArray(maxInputSize * maxOutputSize) }
        val biasesDelta = Array(2) { FloatArray(maxOutputSize * sampleSize) }

        val costFunction = MSECostFunction()
        val costs = FloatArray(maxOutputSize * sampleSize)

        var cost = 0f
        for (i in 0..1000) {
            firstLayer.forwardTraining(input, 0, activations[0], predictions[0], sampleSize)
            secondLayer.forwardTraining(
                predictions[0], 0, activations[1], predictions[1], sampleSize
            )

            cost = costFunction.value(
                predictions[1], 0, target, 0, sampleSize * outputSize, sampleSize
            )
            if (i == 0) {
                println("Initial cost: $cost")
            }

            costFunction.derivative(
                predictions[1], 0, target, 0, costs,
                0, sampleSize * outputSize
            )
            secondLayer.backwardLastLayer(
                predictions[0], activations[0], activations[1], costs, costs, weightsDelta[1],
                biasesDelta[1],
                sampleSize
            )
            firstLayer.backwardZeroLayer(
                input, 0,
                costs, weightsDelta[0], biasesDelta[0], sampleSize
            )

            firstLayer.updateWeightsAndBiases(
                weightsDelta[0], biasesDelta[0],
                alpha, sampleSize
            )
            secondLayer.updateWeightsAndBiases(
                weightsDelta[1], biasesDelta[1],
                alpha, sampleSize
            )

            firstLayer.forwardTraining(input, 0, activations[0], predictions[0], sampleSize)
            secondLayer.forwardTraining(
                predictions[0], 0, activations[1], predictions[1], sampleSize
            )

            val newCost = costFunction.value(
                predictions[1], 0, target, 0, sampleSize * outputSize, sampleSize
            )
            Assertions.assertTrue(
                newCost < cost, "Cost increased: " + newCost + " > " +
                        cost + " on iteration " + i
            )
            cost = newCost
        }

        println("Cost: $cost")
    }

    @Suppress("SameParameterValue")
    private fun nLayersTest(
        inputSize: Int, layersSize: IntArray, outputSize: Int, sampleSize: Int,
        input: FloatArray, target: FloatArray, random: Random
    ) {
        val alpha = 0.0001f
        val layers = arrayOfNulls<DenseLayer>(layersSize.size + 1)

        var maxInputSize = inputSize
        var maxOutputSize = outputSize

        layers[0] = DenseLayer(
            inputSize, layersSize[0], LeakyLeRU(random.nextLong()),
            WeightsOptimizer.OptimizerType.SIMPLE
        )
        for (i in 1 until layers.size - 1) {
            layers[i] = DenseLayer(
                layersSize[i - 1],
                layersSize[i],
                LeakyLeRU(random.nextLong()),
                WeightsOptimizer.OptimizerType.SIMPLE
            )
        }
        layers[layers.size - 1] = DenseLayer(
            layersSize[layersSize.size - 1],
            outputSize,
            LeakyLeRU(random.nextLong()),
            WeightsOptimizer.OptimizerType.SIMPLE
        )

        for (layerSize in layersSize) {
            maxInputSize = max(maxInputSize.toDouble(), layerSize.toDouble()).toInt()
            maxOutputSize = max(maxOutputSize.toDouble(), layerSize.toDouble()).toInt()
        }

        val activations = arrayOfNulls<FloatArray>(layers.size)
        val predictions = arrayOfNulls<FloatArray>(layers.size)

        for (i in layers.indices) {
            activations[i] = FloatArray(sampleSize * layers[i]!!.outputSize)
            predictions[i] = FloatArray(sampleSize * layers[i]!!.outputSize)
        }

        val weightsDelta = Array(layers.size) { FloatArray(maxInputSize * maxOutputSize) }
        val biasesDelta = Array(layers.size) { FloatArray(maxOutputSize * sampleSize) }

        val costFunction = MSECostFunction()
        val costs = FloatArray(maxOutputSize * sampleSize)

        var cost = 0f
        for (i in 0..1000) {
            layers[0]!!
                .forwardTraining(input, 0, activations[0], predictions[0], sampleSize)
            for (n in 1 until layers.size) {
                layers[n]!!.forwardTraining(
                    predictions[n - 1], 0, activations[n], predictions[n], sampleSize
                )
            }

            val lastIndex = layers.size - 1
            cost = costFunction.value(
                predictions[lastIndex], 0, target, 0,
                sampleSize * outputSize, sampleSize
            )
            if (i == 0) {
                println("Initial cost: $cost")
            }

            costFunction.derivative(
                predictions[lastIndex], 0, target, 0, costs, 0,
                sampleSize * outputSize
            )

            layers[lastIndex]!!.backwardLastLayer(
                predictions[lastIndex - 1], activations[lastIndex - 1],
                activations[lastIndex], costs, costs, weightsDelta[lastIndex],
                biasesDelta[lastIndex],
                sampleSize
            )

            for (n in lastIndex - 1 downTo 1) {
                layers[n]!!.backwardMiddleLayer(
                    predictions[n - 1], costs, activations[n - 1], costs, weightsDelta[n],
                    biasesDelta[n],
                    sampleSize
                )
            }

            layers[0]!!.backwardZeroLayer(
                input, 0,
                costs, weightsDelta[0], biasesDelta[0], sampleSize
            )

            for (n in layers.indices) {
                layers[n]!!.updateWeightsAndBiases(
                    weightsDelta[n], biasesDelta[n],
                    alpha, sampleSize
                )
            }

            layers[0]!!.forwardTraining(
                input, 0,
                activations[0], predictions[0], sampleSize
            )
            for (n in 1 until layers.size) {
                layers[n]!!.forwardTraining(
                    predictions[n - 1], 0,
                    activations[n], predictions[n], sampleSize
                )
            }

            val newCost = costFunction.value(
                predictions[lastIndex], 0, target, 0,
                sampleSize * outputSize, sampleSize
            )
            Assertions.assertTrue(
                newCost < cost, "Cost increased: " + newCost + " > " +
                        cost + " on iteration " + i
            )
            cost = newCost
        }

        println("Cost: $cost")
    }
}