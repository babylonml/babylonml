package com.tornadoml.cpu

import org.apache.commons.rng.UniformRandomProvider
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class EmbeddingLayerTests {

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTrainingSingleSampleTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)
        forwardTrainingTest(
            source,
            sampleCount = 1,
            inputSize = source.nextInt(1, 100),
            vocabularySize = source.nextInt(1, 100),
            embeddingDimensions = source.nextInt(1, 100)
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTrainingMultiSampleTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)
        forwardTrainingTest(
            source,
            sampleCount = source.nextInt(2, 10),
            inputSize = source.nextInt(1, 100),
            vocabularySize = source.nextInt(1, 100),
            embeddingDimensions = source.nextInt(1, 100)
        )
    }

    private fun forwardTrainingTest(
        source: UniformRandomProvider,
        sampleCount: Int,
        inputSize: Int,
        vocabularySize: Int,
        embeddingDimensions: Int
    ) {

        val layer = EmbeddingLayer(vocabularySize, embeddingDimensions, inputSize, source.nextLong())
        val lookupTable = FloatMatrix(layer.lookupTable)

        // sampleCounts x inputSize
        val inputs = FloatMatrix(
            Array(sampleCount) { FloatArray(inputSize) { source.nextInt(vocabularySize).toFloat() } }
        )

        // sampleCounts x inputSize x embeddingDimension
        val expectedOutputs = Array(sampleCount) { toOneHot(inputs.data[it], vocabularySize) * lookupTable }

        val activationArg = FloatArray(inputSize * embeddingDimensions * sampleCount)
        val prediction = FloatArray(inputSize * embeddingDimensions * sampleCount)

        // transpose input to place samples in columns
        layer.forwardTraining(inputs.transpose().toFlatArray(), 0, activationArg, prediction, sampleCount)

        Assertions.assertArrayEquals(prediction, activationArg, 0.001f)

        // transpose output to place samples in rows
        val predictionGroupedBySamples =
            FloatMatrix(embeddingDimensions * inputSize, sampleCount, prediction).transpose()

        for (i in 0 until sampleCount) {
            Assertions.assertArrayEquals(expectedOutputs[i].toFlatArray(), predictionGroupedBySamples.data[i], 0.001f)
        }
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun backwardFirstLayerSingleSampleTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)
        backwardFirstLayerTest(
            source,
            sampleCount = 1,
            inputSize = source.nextInt(1, 100),
            vocabularySize = source.nextInt(1, 100),
            embeddingDimensions = source.nextInt(1, 100)
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun backwardFirstLayerMultiSampleTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)
        backwardFirstLayerTest(
            source,
            sampleCount = source.nextInt(2, 10),
            inputSize = source.nextInt(1, 100),
            vocabularySize = source.nextInt(1, 100),
            embeddingDimensions = source.nextInt(1, 100)
        )
    }


    private fun backwardFirstLayerTest(
        source: UniformRandomProvider,
        sampleCount: Int,
        inputSize: Int,
        vocabularySize: Int,
        embeddingDimensions: Int
    ) {
        val layer = EmbeddingLayer(vocabularySize, embeddingDimensions, inputSize, source.nextLong())

        // sampleCounts x inputSize
        val inputs = FloatMatrix(
            Array(sampleCount) { FloatArray(inputSize) { source.nextInt(vocabularySize).toFloat() } }
        )

        val dLdZ = FloatMatrix(inputSize * embeddingDimensions, sampleCount)
        dLdZ.fillRandom(source)

        val weightsDelta = FloatArray(vocabularySize * embeddingDimensions)
        val biasesDelta = FloatArray(inputSize * embeddingDimensions * sampleCount)

        layer.backwardFirstLayer(
            inputs.transpose().toFlatArray(), 0, dLdZ.toFlatArray(), weightsDelta, biasesDelta, sampleCount
        )

        biasesDelta.forEach { Assertions.assertEquals(0.0f, it) }

        val dLdZGroupedBySamples =
            FloatMatrix(inputSize * embeddingDimensions, sampleCount, dLdZ.toFlatArray()).transpose()

        val expectedWeightsDelta = Array(sampleCount) {
            val oneHot = toOneHot(inputs.data[it], vocabularySize)
            val dlDzSample = FloatMatrix(inputSize, embeddingDimensions, dLdZGroupedBySamples.data[it]).transpose()
            dlDzSample * oneHot
        }.reduce { acc, floatMatrix -> acc + floatMatrix }

        Assertions.assertArrayEquals(expectedWeightsDelta.transpose().toFlatArray(), weightsDelta, 0.001f)
    }

    private fun toOneHot(vector: FloatArray, vocabularySize: Int): FloatMatrix {
        val oneHot = FloatMatrix(vector.size, vocabularySize)
        for (i in vector.indices) {
            val v = vector[i]
            assert(v.toInt().toFloat() == v)
            oneHot[i, v.toInt()] = 1.0f
        }
        return oneHot
    }
}
