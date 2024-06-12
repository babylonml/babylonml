package com.tornadoml.cpu

import com.babylonml.FloatVector
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.matrix.FloatMatrix
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class MatrixOperationsTests {
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
            secondMatrixOffset, secondMatrixRows, secondMatrixColumns, result, 0
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
            matrixOffset, matrixRows, matrixColumns, result, 0
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
    fun reduceMatrixToVectorByColumnsTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 100)
        val matrixColumns = source.nextInt(1, 100)

        val matrix = FloatMatrix(matrixRows, matrixColumns)
        matrix.fillRandom(source)

        val result = FloatArray(matrixRows + 1) {
            source.nextFloat()
        }

        val matrixArray = FloatArray(matrixRows * matrixColumns + 1) {
            source.nextFloat()
        }
        MatrixOperations.reduceMatrixToVectorByColumns(
            matrix.toFlatArray().copyInto(matrixArray), 0, matrixRows, matrixColumns,
            result, 0
        )

        Assertions.assertArrayEquals(matrix.reduceByColumns().toArray(), result.copyOfRange(0, matrixRows), 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun reduceMatrixToVectorByRowsTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 100)
        val matrixColumns = source.nextInt(1, 100)

        val matrix = FloatMatrix(matrixRows, matrixColumns)
        matrix.fillRandom(source)

        val matrixOffset = source.nextInt(17)
        val vectorOffset = source.nextInt(17)

        val result = FloatArray(matrixColumns + 1 + vectorOffset) {
            source.nextFloat()
        }

        val matrixArray = FloatArray(matrixRows * matrixColumns + 1 + matrixOffset) {
            source.nextFloat()
        }
        MatrixOperations.reduceMatrixToVectorByRows(
            matrix.toFlatArray().copyInto(matrixArray, matrixOffset), matrixOffset, matrixRows, matrixColumns,
            result, vectorOffset
        )

        Assertions.assertArrayEquals(
            matrix.reduceByRows().toArray(),
            result.copyOfRange(vectorOffset, matrixColumns + vectorOffset), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun broadCastVectorToMatrixByColumnsTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val vectorLength = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val vector = FloatVector(vectorLength)
        val matrix = FloatArray(vectorLength * columns + 1) {
            source.nextFloat()
        }

        vector.fillRandom(source)

        val vectorArray = FloatArray(vectorLength + 1) {
            source.nextFloat()
        }

        MatrixOperations.broadcastVectorToMatrixByColumns(
            vector.toArray().copyInto(vectorArray), 0, matrix, 0,
            vectorLength, columns
        )
        Assertions.assertArrayEquals(
            vector.broadcastColumns(columns).toFlatArray(),
            matrix.copyOfRange(0, vectorLength * columns),
            0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun broadCastVectorToMatrixByRowsTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val vectorLength = source.nextInt(1, 100)
        val rows = source.nextInt(1, 100)

        val vector = FloatVector(vectorLength)

        val vectorOffset = source.nextInt(17)
        val matrixOffset = source.nextInt(17)

        val matrix = FloatArray(rows * vectorLength + 1 + matrixOffset) {
            source.nextFloat()
        }
        vector.fillRandom(source)

        val vectorArray = FloatArray(vectorLength + 1 + vectorOffset) {
            source.nextFloat()
        }

        MatrixOperations.broadcastVectorToMatrixByRows(
            vector.toArray().copyInto(vectorArray, vectorOffset),
            vectorOffset, matrix, matrixOffset,
            rows, vectorLength
        )

        Assertions.assertArrayEquals(
            vector.broadcastRows(rows).toFlatArray(),
            matrix.copyOfRange(matrixOffset, matrixOffset + rows * vectorLength),
            0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun subMatrixTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 1000)
        val matrixColumns = source.nextInt(2, 1000)

        val matrix = FloatMatrix(matrixRows, matrixColumns)
        matrix.fillRandom(source)

        val startColumn = source.nextInt(matrixColumns - 1)
        val columns = source.nextInt(matrixColumns - startColumn)

        val copy = matrix.subColumns(startColumn, columns)

        val result = FloatArray(matrixRows * columns) {
            source.nextFloat()
        }

        MatrixOperations.subMatrix(
            matrix.toFlatArray(), startColumn, matrixRows, matrixColumns,
            result, columns
        )

        Assertions.assertArrayEquals(copy.toFlatArray(), result, 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun softMaxByColumnsValueCalculationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 100)
        val matrixColumns = source.nextInt(1, 100)

        val matrix = FloatMatrix(matrixRows, matrixColumns)
        matrix.fillRandom(source, -10.0f, 10.0f)

        val expected = matrix.softMaxByColumns()
        val actual = FloatArray(matrixRows * matrixColumns) {
            source.nextFloat()
        }

        MatrixOperations.softMaxByColumns(
            matrix.toFlatArray(), 0, matrixRows, matrixColumns, actual, 0
        )

        Assertions.assertArrayEquals(expected.toFlatArray(), actual, 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun softMaxByColumnsIsFiniteTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 100)
        val matrixColumns = source.nextInt(1, 100)

        val matrix = FloatMatrix(matrixRows, matrixColumns)
        matrix.fillRandom(source, -500f, 500f)

        val actual = FloatArray(matrixRows * matrixColumns) {
            source.nextFloat()
        }

        MatrixOperations.softMaxByColumns(
            matrix.toFlatArray(), 0, matrixRows, matrixColumns, actual, 0
        )

        for (i in 0 until matrixRows * matrixColumns) {
            Assertions.assertTrue(actual[i].isFinite())
        }
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun softMaxByRowsValueCalculationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 100)
        val matrixColumns = source.nextInt(1, 100)

        val matrix = FloatMatrix(matrixRows, matrixColumns)
        matrix.fillRandom(source, -10.0f, 10.0f)

        val expected = matrix.softMaxByRows()

        val matrixOffset = source.nextInt(17)
        val resultOffset = source.nextInt(17)

        val actual = FloatArray(matrixRows * matrixColumns + resultOffset) {
            source.nextFloat()
        }

        MatrixOperations.softMaxByRows(
            matrix.toFlatArray().copyInto(FloatArray(matrix.size + matrixOffset), matrixOffset),
            matrixOffset, matrixRows, matrixColumns, actual, resultOffset
        )

        Assertions.assertArrayEquals(expected.toFlatArray(), actual.copyOfRange(resultOffset, actual.size), 0.001f)
    }
}