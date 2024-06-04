package com.tornadoml.cpu

import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class SoftMaxByRowsTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun softMaxPredictTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 50)
        val samplesCount = source.nextInt(1, 50)

        val input  = FloatMatrix(inputSize, samplesCount)
        input.fillRandom(source)
        val expected = input.softMaxByColumns()

        val softMaxLayer = SoftMaxLayer(inputSize)
        val actual = FloatArray(inputSize * samplesCount) {
            source.nextFloat()
        }

        softMaxLayer.predict(input.toFlatArray(), actual, samplesCount)
        Assertions.assertArrayEquals(expected.toFlatArray(), actual, 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun layerErrorTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 50)
        val samplesCount = source.nextInt(1, 50)

        val input  = FloatMatrix(inputSize, samplesCount)
        input.fillRandom(source)
        val expected = FloatMatrix(inputSize, samplesCount)
        expected.fillRandom(source)

        val expectedResult = input - expected
        val actualResult = FloatArray(inputSize * samplesCount) {
            source.nextFloat()
        }
        val softMaxLayer = SoftMaxLayer(inputSize)
        softMaxLayer.backwardLastLayer(input.toFlatArray(), expected.toFlatArray(), actualResult, samplesCount)

        Assertions.assertArrayEquals(expectedResult.toFlatArray(), actualResult, 0.001f)
    }
}