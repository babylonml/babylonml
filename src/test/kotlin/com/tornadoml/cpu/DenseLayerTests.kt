package com.tornadoml.cpu

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import com.tornadoml.mnist.MNISTLoader
import java.util.*
import kotlin.math.max

class DenseLayerTests {
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

            layer.updateWeightsAndBiases(weightsDelta, biasesDelta, 0.01f, sampleSize)

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
                predictions[0], activations[0], activations[1], costs, weightsDelta[1], biasesDelta[1],
                sampleSize
            )
            firstLayer.backwardZeroLayer(
                input, 0,
                costs, weightsDelta[0], biasesDelta[0], sampleSize
            )

            firstLayer.updateWeightsAndBiases(
                weightsDelta[0], biasesDelta[0],
                0.01f, sampleSize
            )
            secondLayer.updateWeightsAndBiases(
                weightsDelta[1], biasesDelta[1],
                0.01f, sampleSize
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
                activations[lastIndex], costs, weightsDelta[lastIndex], biasesDelta[lastIndex],
                sampleSize
            )

            for (n in lastIndex - 1 downTo 1) {
                layers[n]!!.backwardMiddleLayer(
                    predictions[n - 1], costs, activations[n - 1],
                    weightsDelta[n], biasesDelta[n],
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
                    0.001f, sampleSize
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