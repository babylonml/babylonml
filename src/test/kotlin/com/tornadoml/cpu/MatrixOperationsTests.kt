package com.tornadoml.cpu

import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource
import java.nio.ByteBuffer
import java.security.SecureRandom

class MatrixOperationsTests {
    private val securesRandom = SecureRandom()

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun matrixMulTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val firsMatrixRows = source.nextInt(1000)
        val firstMatrixColumns = source.nextInt(1000)

        val firstMatrixOffset = source.nextInt(17)
        val secondMatrixOffset = source.nextInt(17)

        val secondMatrixRows = firstMatrixColumns
        val secondMatrixColumns = source.nextInt(1000)

        val firstMatrixArray = FloatArray(firsMatrixRows * firstMatrixColumns + firstMatrixOffset + 1) {
            source.nextFloat()
        }
        val secondMatrixArray = FloatArray(secondMatrixRows * secondMatrixColumns + secondMatrixOffset + 1) {
            source.nextFloat()
        }

        val firstMatrix = FloatMatrix(firsMatrixRows, firstMatrixColumns)
        val secondMatrix = FloatMatrix(secondMatrixRows, secondMatrixColumns)

        firstMatrix.fillRandom(source)
        secondMatrix.fillRandom(source)

        val result = FloatArray(firsMatrixRows * secondMatrixColumns + 1) {
            source.nextFloat()
        }

        MatrixOperations.matrixToMatrixMultiplication(
            firstMatrix.toFlatArray().copyInto(firstMatrixArray, firstMatrixOffset),
            firstMatrixOffset, firsMatrixRows, firstMatrixColumns,
            secondMatrix.toFlatArray().copyInto(secondMatrixArray, secondMatrixOffset),
            secondMatrixOffset, secondMatrixRows, secondMatrixColumns, result
        )

        Assertions.assertArrayEquals(
            (firstMatrix * secondMatrix).toFlatArray(),
            result.copyOfRange(0, firsMatrixRows * secondMatrixColumns), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun matrixTransposeTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1000)
        val matrixColumns = source.nextInt(1000)

        val matrixOffset = source.nextInt(17)
        val matrixArray = FloatArray(matrixRows * matrixColumns + matrixOffset + 1) {
            source.nextFloat()
        }

        val matrix = FloatMatrix(matrixRows, matrixColumns)
        matrix.fillRandom(source)

        val result = FloatArray(matrixRows * matrixColumns + 1) {
            source.nextFloat()
        }

        MatrixOperations.transposeMatrix(
            matrix.toFlatArray().copyInto(matrixArray, matrixOffset),
            matrixOffset, matrixRows, matrixColumns, result
        )

        Assertions.assertArrayEquals(
            matrix.transpose().toFlatArray(), result.copyOfRange(
                0,
                matrixRows * matrixColumns
            ), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun reduceMatrixToVectorTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1000)
        val matrixColumns = source.nextInt(1000)

        val matrix = FloatMatrix(matrixRows, matrixColumns)
        matrix.fillRandom(source)

        val result = FloatArray(matrixRows + 1) {
            source.nextFloat()
        }

        val matrixArray = FloatArray(matrixRows * matrixColumns + 1) {
            source.nextFloat()
        }
        MatrixOperations.reduceMatrixToVector(
            matrix.toFlatArray().copyInto(matrixArray), matrixRows, matrixColumns,
            result
        )

        Assertions.assertArrayEquals(matrix.reduce().toArray(), result.copyOfRange(0, matrixRows), 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun broadCastVectorToMatrixTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val vectorLength = source.nextInt(1000)
        val columns = source.nextInt(1000)

        val vector = FloatVector(vectorLength)
        val matrix = FloatArray(vectorLength * columns + 1) {
            source.nextFloat()
        }

        vector.fillRandom(source)

        val vectorArray = FloatArray(vectorLength + 1) {
            source.nextFloat()
        }

        MatrixOperations.broadcastVectorToMatrix(
            vector.toArray().copyInto(vectorArray), matrix,
            vectorLength, columns
        )
        Assertions.assertArrayEquals(
            vector.broadcast(columns).toFlatArray(),
            matrix.copyOfRange(0, vectorLength * columns),
            0.001f
        )
    }
}