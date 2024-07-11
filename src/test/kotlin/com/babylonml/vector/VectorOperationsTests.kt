package com.babylonml.vector

import com.babylonml.backend.cpu.VectorOperations
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import java.nio.ByteBuffer
import java.security.SecureRandom

class VectorOperationsTests {
    private val securesRandom = SecureRandom()

    fun multiplyVectorToScalarTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("multiplyVectorToScalarTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val vectorLength = source.nextInt(1000)
            val scalar = source.nextFloat()

            val vector = FloatVector(vectorLength)
            vector.fillRandom(source)

            val vectorOffset = source.nextInt(17)
            val vectorArray = FloatArray(vectorOffset + vectorLength + 1) {
                source.nextFloat()
            }

            val resultOffset = source.nextInt(17)
            val result = FloatArray(resultOffset + vectorLength + 1) {
                source.nextFloat()
            }

            VectorOperations.multiplyVectorToScalar(
                vector.toArray().copyInto(vectorArray, vectorOffset), vectorOffset,
                scalar,
                result,
                resultOffset,
                vectorLength
            )

            val resultArray = FloatArray(vectorLength + 1) {
                result[it + resultOffset]
            }

            Assertions.assertArrayEquals(
                (vector * scalar).toArray(), resultArray.copyOfRange(0, vectorLength),
                0.001f
            )
        }
    }

    fun addVectorToVectorTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("addVectorToVectorTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val vectorLength = source.nextInt(1000)

            val first = FloatVector(vectorLength)
            val second = FloatVector(vectorLength)

            first.fillRandom(source)
            second.fillRandom(source)

            val firstArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            val secondArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }

            val result = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }

            VectorOperations.addVectorToVector(
                first.toArray().copyInto(firstArray), 0,
                second.toArray().copyInto(secondArray), 0,
                result, 0,
                vectorLength
            )

            Assertions.assertArrayEquals((first + second).toArray(), result.copyOfRange(0, vectorLength), 0.001f)
        }
    }

    fun vectorToVectorElementWiseMultiplicationTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("vectorToVectorElementWiseMultiplicationTest seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            val vectorLength = source.nextInt(1000)

            val first = FloatVector(vectorLength)
            val second = FloatVector(vectorLength)

            first.fillRandom(source)
            second.fillRandom(source)

            val result = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }

            val firstArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            val secondArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }

            VectorOperations.vectorToVectorElementWiseMultiplication(
                first.toArray().copyInto(firstArray), 0,
                second.toArray().copyInto(secondArray), 0,
                result, 0,
                vectorLength
            )

            Assertions.assertArrayEquals((first * second).toArray(), result.copyOfRange(0, vectorLength), 0.001f)
        }
    }

    fun divideScalarOnVectorElementsTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("divideScalarOnVectorElementsTest seed: $seed")

            val source = RandomSource.ISAAC.create(seed)

            val vectorLength = source.nextInt(1000)
            val scalar = source.nextFloat()

            val vector = FloatVector(vectorLength)

            val result = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            vector.fillRandom(source)

            val vectorArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }

            VectorOperations.divideScalarOnVectorElements(
                scalar,
                vector.toArray().copyInto(vectorArray), 0,
                result, 0,
                vectorLength
            )

            Assertions.assertArrayEquals((scalar / vector).toArray(), result.copyOfRange(0, vectorLength), 0.001f)
        }
    }

    fun sqrtEachElementOfVectorTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("sqrtEachElementOfVectorTest seed: $seed")

            val source = RandomSource.ISAAC.create(seed)
            val vectorLength = source.nextInt(1000)

            val vector = FloatVector(vectorLength)

            val result = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }

            vector.fillRandom(source)

            val vectorArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            VectorOperations.vectorElementsSqrt(
                vector.toArray().copyInto(vectorArray), 0,
                result,
                0,
                vectorLength
            )
            Assertions.assertArrayEquals(vector.sqrt().toArray(), result.copyOfRange(0, vectorLength), 0.001f)
        }
    }

    fun vectorElementsExpTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("vectorElementsExpTest seed: $seed")

            val source = RandomSource.ISAAC.create(seed)
            val vectorLength = source.nextInt(1000)

            val vector = FloatVector(vectorLength)

            val result = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            vector.fillRandom(source)

            val vectorArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            VectorOperations.vectorElementsExp(
                vector.toArray().copyInto(vectorArray),
                result, vectorLength
            )
            Assertions.assertArrayEquals(vector.exp().toArray(), result.copyOfRange(0, vectorLength), 0.001f)
        }
    }

    fun maxBetweenVectorElementsTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("maxBetweenVectorElementsTest seed: $seed")

            val source = RandomSource.ISAAC.create(seed)
            val vectorLength = source.nextInt(1000)

            val firstVector = FloatVector(vectorLength)
            val secondVector = FloatVector(vectorLength)

            firstVector.fillRandom(source)
            secondVector.fillRandom(source)

            val result = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }

            val firstArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            val secondArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            VectorOperations.maxBetweenVectorElements(
                firstVector.toArray().copyInto(firstArray), 0,
                secondVector.toArray().copyInto(secondArray), 0,
                result, 0,
                vectorLength
            )
            Assertions.assertArrayEquals(
                firstVector.max(secondVector).toArray(),
                result.copyOfRange(0, vectorLength),
                0.001f
            )
        }
    }

    fun sumVectorElementsTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("sumVectorElementsTest seed: $seed")

            val source = RandomSource.ISAAC.create(seed)
            val vectorLength = source.nextInt(1000)

            val vector = FloatVector(vectorLength)
            vector.fillRandom(source)

            val vectorArrayOffset = source.nextInt(17)
            val vectorArray = FloatArray(vectorArrayOffset + vectorLength + 1) {
                source.nextFloat()
            }

            val result = VectorOperations.sumVectorElements(
                vector.toArray().copyInto(vectorArray, vectorArrayOffset),
                vectorArrayOffset, vectorLength
            )
            Assertions.assertEquals(vector.sum(), result, 0.001f)
        }
    }

    fun addScalarToVectorTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("addScalarToVectorTest seed: $seed")

            val source = RandomSource.ISAAC.create(seed)
            val vectorLength = source.nextInt(1000)
            val scalar = source.nextFloat()

            val vector = FloatVector(vectorLength)
            vector.fillRandom(source)


            val result = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            val vectorArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }

            VectorOperations.addScalarToVector(
                scalar,
                vector.toArray().copyInto(vectorArray), 0,
                result, 0, vectorLength
            )

            Assertions.assertArrayEquals((vector + scalar).toArray(), result.copyOfRange(0, vectorLength), 0.001f)
        }
    }

    fun subtractVectorFromVectorTest() {
        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("subtractVectorFromVectorTest seed: $seed")

            val source = RandomSource.ISAAC.create(seed)
            val vectorLength = source.nextInt(1000)

            val first = FloatVector(vectorLength)
            val second = FloatVector(vectorLength)

            first.fillRandom(source)
            second.fillRandom(source)

            val result = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            val firstArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            val secondArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }

            VectorOperations.subtractVectorFromVector(
                first.toArray().copyInto(firstArray), 0,
                second.toArray().copyInto(secondArray), 0,
                result, 0,
                vectorLength
            )

            Assertions.assertArrayEquals((first - second).toArray(), result.copyOfRange(0, vectorLength), 0.001f)
        }
    }
}