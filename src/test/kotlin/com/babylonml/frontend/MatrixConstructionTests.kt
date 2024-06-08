package com.babylonml.frontend

import com.tornadoml.cpu.SeedsArgumentsProvider
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class MatrixConstructionTests {

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testZeros(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val matrix = Matrix.zeros(rows, columns)

        assertEquals(MatrixDims(rows, columns), matrix.dims)
        assertArrayEquals(FloatArray(rows * columns) { 0f }, matrix.data)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testOnes(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val matrix = Matrix.ones(rows, columns)

        assertEquals(MatrixDims(rows, columns), matrix.dims)
        assertArrayEquals(FloatArray(rows * columns) { 1f }, matrix.data)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testFill(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)
        val value = source.nextFloat()

        val matrix = Matrix.fill(rows, columns, value)

        assertEquals(MatrixDims(rows, columns), matrix.dims)
        assertArrayEquals(FloatArray(rows * columns) { value }, matrix.data)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testFillFunction(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val randomData = FloatArray(rows * columns) { source.nextFloat() }

        val matrix = Matrix.fill(rows, columns) { i, j -> randomData[i * columns + j] }

        assertEquals(MatrixDims(rows, columns), matrix.dims)
        assertArrayEquals(randomData, matrix.data)
    }


    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testIdentity(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val size = source.nextInt(1, 100)

        val matrix = Matrix.identity(size)

        assertEquals(MatrixDims(size, size), matrix.dims)
        assertArrayEquals(FloatArray(size * size) { i -> if (i % (size + 1) == 0) 1f else 0f }, matrix.data)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testFrom2DArray(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val array = Array(rows) { FloatArray(columns) { source.nextFloat() } }

        val matrix = Matrix.from2DArray(array)

        assertEquals(MatrixDims(rows, columns), matrix.dims)
        assertArrayEquals(FloatArray(rows * columns) { i -> array[i / columns][i % columns] }, matrix.data)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testFromFlatArray(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val flatArray = FloatArray(rows * columns) { source.nextFloat() }

        val matrix = Matrix.fromFlatArray(rows, columns, flatArray)

        assertEquals(MatrixDims(rows, columns), matrix.dims)
        assertArrayEquals(flatArray, matrix.data)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testRowVector(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val columns = source.nextInt(1, 100)

        val data = FloatArray(columns) { source.nextFloat() }

        val matrix = Matrix.rowVector(*data)

        assertEquals(MatrixDims(1, columns), matrix.dims)
        assertArrayEquals(data, matrix.data)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testColumnVector(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)

        val data = FloatArray(rows) { source.nextFloat() }

        val matrix = Matrix.columnVector(*data)

        assertEquals(MatrixDims(rows, 1), matrix.dims)
        assertArrayEquals(data, matrix.data)
    }
}
