package com.babylonml.tensor

import com.babylonml.SeedsArgumentsProvider
import com.babylonml.backend.cpu.TensorOperations
import org.apache.commons.rng.sampling.PermutationSampler
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class TensorOperationsTest {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun vectorToMatrixByRowsBroadcastTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val vectorSize = source.nextInt(1, 100)
        val vector = FloatTensor.random(source, vectorSize)

        val matrixRows = source.nextInt(1, 100)
        val matrix = vector.broadcast(matrixRows, vectorSize)

        val actualResultArray = FloatArray(matrixRows * vectorSize)
        TensorOperations.broadcast(
            vector.toFlatArray(), 0, intArrayOf(vectorSize),
            actualResultArray, 0, intArrayOf(matrixRows, vectorSize)
        )

        Assertions.assertArrayEquals(
            matrix.toFlatArray(),
            actualResultArray, 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun matrixToMatrixByColumnsBroadcastTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 100)
        val initialMatrix = FloatTensor.random(source, matrixRows, 1)

        val matrixColumns = source.nextInt(1, 100)
        val expectedResultMatrix = initialMatrix.broadcast(matrixRows, matrixColumns)

        val actualResultArray = FloatArray(matrixColumns * matrixRows)

        TensorOperations.broadcast(
            initialMatrix.toFlatArray(), 0, intArrayOf(matrixRows, 1),
            actualResultArray, 0, intArrayOf(matrixRows, matrixColumns)
        )

        Assertions.assertArrayEquals(expectedResultMatrix.toFlatArray(), actualResultArray, 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun scalarToMatrixBroadcastTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 100)
        val initialMatrix = FloatTensor.random(source, 1, 1)

        val matrixColumns = source.nextInt(1, 100)
        val expectedResultMatrix = initialMatrix.broadcast(matrixRows, matrixColumns)

        val actualResultArray = FloatArray(matrixColumns * matrixRows)

        TensorOperations.broadcast(
            initialMatrix.toFlatArray(), 0, intArrayOf(1, 1),
            actualResultArray, 0, intArrayOf(matrixRows, matrixColumns)
        )

        Assertions.assertArrayEquals(expectedResultMatrix.toFlatArray(), actualResultArray, 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun multipleDimensionsBroadcastTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val tensorDimensions = source.nextInt(3, 10)
        val newShape = IntArray(tensorDimensions) { source.nextInt(2, 10) }

        val broadcastDimensionsCount = source.nextInt(1, tensorDimensions)
        val permutation = PermutationSampler.natural(tensorDimensions)
        PermutationSampler.shuffle(source, permutation)

        val shape = newShape.copyOf()
        for (i in 0 until broadcastDimensionsCount) {
            shape[permutation[i]] = 1
        }

        val tensor = FloatTensor.random(source, *shape)
        val broadcastTensor = tensor.broadcast(*newShape)
        val actualResultArray = FloatArray(broadcastTensor.size)

        TensorOperations.broadcast(
            tensor.toFlatArray(), 0, shape,
            actualResultArray, 0, newShape
        )

        Assertions.assertArrayEquals(broadcastTensor.toFlatArray(), actualResultArray, 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun vectorToMatrixByRowsReduceTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val vectorSize = source.nextInt(1, 100)
        val matrixColumns = source.nextInt(1, 100)

        val matrix = FloatTensor.random(source, vectorSize, matrixColumns)

        val vector = matrix.reduce(matrixColumns)
        val actualResultArray = FloatArray(matrixColumns)

        TensorOperations.reduce(
            matrix.toFlatArray(), 0, intArrayOf(vectorSize, matrixColumns),
            actualResultArray, 0, intArrayOf(matrixColumns)
        )

        Assertions.assertArrayEquals(
            vector.toFlatArray(),
            actualResultArray, 0.001f
        )
    }
}