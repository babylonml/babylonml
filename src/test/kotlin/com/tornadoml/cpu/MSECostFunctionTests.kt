package com.tornadoml.cpu

import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import java.nio.ByteBuffer
import java.security.SecureRandom

class MSECostFunctionTests {
    private val securesRandom = SecureRandom()

    @Test
    fun calculationTest() {
        val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
        val seed = bBuffer.getLong()

        println("calculationTest seed: $seed")
        val source = RandomSource.ISAAC.create(seed)

        val target = FloatMatrix(source.nextInt(10, 100), source.nextInt(10, 100))
        target.fillRandom(source)

        val actual = FloatMatrix(target.rows, target.cols)
        actual.fillRandom(source)

        val actualArray = FloatArray(actual.size + 1) {
            source.nextFloat()
        }

        val targetArray = FloatArray(actual.size + 3) {
            source.nextFloat()
        }

        val result = MSECostFunction().value(
            actual.toFlatArray().copyInto(actualArray, 1), 1,
            target.toFlatArray().copyInto(targetArray, 3), 3,
            target.size, target.cols
        )

        Assertions.assertEquals(mseCostFunction(actual, target), result, 0.0001f)
    }

    @Test
    fun derivativeTest() {
        val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
        val seed = bBuffer.getLong()

        println("derivativeTest seed: $seed")
        val source = RandomSource.ISAAC.create(seed)

        val target = FloatMatrix(source.nextInt(10, 100), source.nextInt(10, 100))
        target.fillRandom(source)

        val actual = FloatMatrix(target.rows, target.cols)
        actual.fillRandom(source)

        val actualArray = FloatArray(actual.size + 1) {
            source.nextFloat()
        }

        val targetArray = FloatArray(actual.size + 3) {
            source.nextFloat()
        }

        val result = mseCostFunctionDerivative(actual, target)

        val resultArray = FloatArray(result.size + 1) {
            source.nextFloat()
        }

        MSECostFunction().derivative(
            actual.toFlatArray().copyInto(actualArray, 1), 1,
            target.toFlatArray().copyInto(targetArray, 3), 3,
            resultArray, 1, target.size
        )

        Assertions.assertArrayEquals(result.toFlatArray(), resultArray.copyOfRange(1, target.size + 1), 0.0001f)
    }
}