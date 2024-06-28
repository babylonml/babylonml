package com.babylonml.tensor

import com.babylonml.SeedsArgumentsProvider
import com.babylonml.backend.cpu.TensorOperations
import it.unimi.dsi.fastutil.ints.IntImmutableList
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

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(vector.size + inputOffset)
        vector.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(matrixRows * vectorSize + outputOffset)

        TensorOperations.broadcast(
            inputArray, inputOffset, IntImmutableList.of(vectorSize),
            outputArray, outputOffset, IntImmutableList.of(matrixRows, vectorSize), -1
        )

        Assertions.assertArrayEquals(
            matrix.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun matrixToMatrixByOnlyRowsBroadcastTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)


        val columns = source.nextInt(1, 100)
        val matrix = FloatTensor.random(source, 1, columns)

        val rows = source.nextInt(1, 100)
        val newColumns = source.nextInt(1, 100)

        val newShape = intArrayOf(rows, newColumns)

        val expectedMatrix = matrix.broadcast(*newShape, till = 1)

        val inputOffset = source.nextInt(8)
        val inputArray = FloatArray(matrix.size + inputOffset)

        matrix.toFlatArray().copyInto(inputArray, inputOffset)

        val outputOffset = source.nextInt(8)
        val outputArray = FloatArray(expectedMatrix.size + outputOffset)

        TensorOperations.broadcast(
            inputArray, inputOffset, IntImmutableList.of(*matrix.shape),
            outputArray, outputOffset, IntImmutableList.of(*newShape), 1
        )

        Assertions.assertArrayEquals(
            expectedMatrix.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size), 0.001f
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

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(initialMatrix.size + inputOffset)
        initialMatrix.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(expectedResultMatrix.size + outputOffset)

        TensorOperations.broadcast(
            inputArray, inputOffset, IntImmutableList.of(matrixRows, 1),
            outputArray, outputOffset, IntImmutableList.of(matrixRows, matrixColumns), -1
        )

        Assertions.assertArrayEquals(
            expectedResultMatrix.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size),
            0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun scalarToMatrixBroadcastTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 100)
        val initialMatrix = FloatTensor.random(source, 1, 1)

        val matrixColumns = source.nextInt(1, 100)
        val expectedResultMatrix = initialMatrix.broadcast(matrixRows, matrixColumns)

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(initialMatrix.size + inputOffset)
        initialMatrix.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(matrixColumns * matrixRows + outputOffset)

        TensorOperations.broadcast(
            inputArray, inputOffset, IntImmutableList.of(1, 1),
            outputArray, outputOffset, IntImmutableList.of(matrixRows, matrixColumns), -1
        )

        Assertions.assertArrayEquals(
            expectedResultMatrix.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun scalarToMatrixBroadcastOnlyRowsTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 100)
        val initialMatrix = FloatTensor.random(source, 1, 1)

        val matrixColumns = source.nextInt(1, 100)
        val newShape = intArrayOf(matrixRows, matrixColumns)

        val expectedMatrix = initialMatrix.broadcast(*newShape, till = 1)

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(initialMatrix.size + inputOffset)
        initialMatrix.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(expectedMatrix.size + outputOffset)

        TensorOperations.broadcast(
            inputArray, inputOffset, IntImmutableList.of(1, 1),
            outputArray, outputOffset, IntImmutableList.of(*newShape), 1
        )

        Assertions.assertArrayEquals(
            expectedMatrix.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun multipleDimensionsBroadcastTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val tensorDimensions = source.nextInt(3, 7)
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

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(tensor.size + inputOffset)
        tensor.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(broadcastTensor.size + outputOffset)

        TensorOperations.broadcast(
            inputArray, inputOffset, IntImmutableList.of(*shape),
            outputArray, outputOffset, IntImmutableList.of(*newShape), -1
        )

        Assertions.assertArrayEquals(
            broadcastTensor.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun multipleDimensionsLimitedRankBroadcastTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val tensorDimensions = source.nextInt(3, 7)
        val newShape = IntArray(tensorDimensions) { source.nextInt(2, 10) }

        val broadcastDimensionsCount = source.nextInt(1, tensorDimensions)
        val permutation = PermutationSampler.natural(tensorDimensions)
        PermutationSampler.shuffle(source, permutation)

        val shape = newShape.copyOf()
        for (i in 0 until broadcastDimensionsCount) {
            shape[permutation[i]] = 1
        }

        val broadcastTillRank = source.nextInt(1, tensorDimensions)

        val tensor = FloatTensor.random(source, *shape)
        val broadcastTensor = tensor.broadcast(*newShape, till = broadcastTillRank)

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(tensor.size + inputOffset)
        tensor.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(broadcastTensor.size + outputOffset)

        TensorOperations.broadcast(
            inputArray, inputOffset, IntImmutableList.of(*shape),
            outputArray, outputOffset, IntImmutableList.of(*newShape), broadcastTillRank
        )

        Assertions.assertArrayEquals(
            broadcastTensor.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun multipleDimensionsCountBroadcastTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val tensorDimensions = source.nextInt(3, 7)
        val newTensorDimensions = source.nextInt(tensorDimensions, tensorDimensions + 4)

        val newShape = IntArray(newTensorDimensions) { source.nextInt(2, 7) }
        val shape = IntArray(tensorDimensions) { newShape[newTensorDimensions - tensorDimensions + it] }

        val broadcastDimensionsCount = source.nextInt(1, tensorDimensions)
        val permutation = PermutationSampler.natural(tensorDimensions)
        PermutationSampler.shuffle(source, permutation)

        for (i in 0 until broadcastDimensionsCount) {
            shape[permutation[i]] = 1
        }

        val tensor = FloatTensor.random(source, *shape)
        val broadcastTensor = tensor.broadcast(*newShape)

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(tensor.size + inputOffset)
        tensor.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(broadcastTensor.size + outputOffset)

        TensorOperations.broadcast(
            inputArray, inputOffset, IntImmutableList.of(*shape),
            outputArray, outputOffset, IntImmutableList.of(*newShape), -1
        )

        Assertions.assertArrayEquals(
            broadcastTensor.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun multipleDimensionsCountBroadcastTillRankTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val tensorDimensions = source.nextInt(3, 7)
        val newTensorDimensions = source.nextInt(tensorDimensions, tensorDimensions + 4)

        val newShape = IntArray(newTensorDimensions) { source.nextInt(2, 7) }
        val shape = IntArray(tensorDimensions) { newShape[newTensorDimensions - tensorDimensions + it] }

        val broadcastDimensionsCount = source.nextInt(1, tensorDimensions)
        val permutation = PermutationSampler.natural(tensorDimensions)
        PermutationSampler.shuffle(source, permutation)

        for (i in 0 until broadcastDimensionsCount) {
            shape[permutation[i]] = 1
        }

        val broadcastTillRank = source.nextInt(1, tensorDimensions)
        val tensor = FloatTensor.random(source, *shape)
        val broadcastTensor = tensor.broadcast(*newShape, till = broadcastTillRank)

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(tensor.size + inputOffset)
        tensor.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(broadcastTensor.size + outputOffset)

        TensorOperations.broadcast(
            inputArray, inputOffset, IntImmutableList.of(*shape),
            outputArray, outputOffset, IntImmutableList.of(*newShape), broadcastTillRank
        )

        Assertions.assertArrayEquals(
            broadcastTensor.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun vectorToMatrixByRowsReduceTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val vectorSize = source.nextInt(1, 100)
        val matrixColumns = source.nextInt(1, 100)

        val matrix = FloatTensor.random(source, vectorSize, matrixColumns)

        val vector = matrix.reduce(matrixColumns)

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(vectorSize * matrixColumns + inputOffset)
        matrix.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(matrixColumns + outputOffset)

        TensorOperations.reduce(
            inputArray, inputOffset, IntImmutableList.of(vectorSize, matrixColumns),
            outputArray, outputOffset, IntImmutableList.of(matrixColumns)
        )

        Assertions.assertArrayEquals(
            vector.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun matrixToMatrixByColumnsReduceTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 100)
        val matrixColumns = source.nextInt(1, 100)

        val matrix = FloatTensor.random(source, matrixRows, matrixColumns)

        val vector = matrix.reduce(matrixRows, 1)

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(matrixRows * matrixColumns + inputOffset)
        matrix.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(matrixRows + outputOffset)

        TensorOperations.reduce(
            inputArray, inputOffset, IntImmutableList.of(matrixRows, matrixColumns),
            outputArray, outputOffset, IntImmutableList.of(matrixRows, 1)
        )

        Assertions.assertArrayEquals(
            vector.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun matrixToScalarReduceTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 100)
        val matrixColumns = source.nextInt(1, 100)

        val initialMatrix = FloatTensor.random(source, matrixRows, matrixColumns)
        val scalar = initialMatrix.reduce(1)

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(matrixRows * matrixColumns + inputOffset)
        initialMatrix.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(1 + outputOffset)

        TensorOperations.reduce(
            inputArray, inputOffset, IntImmutableList.of(matrixRows, matrixColumns),
            outputArray, outputOffset, IntImmutableList.of(1)
        )

        Assertions.assertEquals(scalar.toFlatArray()[0], outputArray[outputOffset], 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun multipleDimensionsReduceTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val tensorDimensions = source.nextInt(3, 6)
        val shape = IntArray(tensorDimensions) { source.nextInt(2, 7) }
        val reduceDimensionsCount = source.nextInt(1, tensorDimensions)
        val permutation = PermutationSampler.natural(tensorDimensions)
        PermutationSampler.shuffle(source, permutation)

        val newShape = shape.copyOf()
        for (i in 0 until reduceDimensionsCount) {
            newShape[permutation[i]] = 1
        }

        val tensor = FloatTensor.random(source, *shape)
        val reducedTensor = tensor.reduce(*newShape)

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(tensor.size + inputOffset)
        tensor.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(reducedTensor.size + outputOffset)

        TensorOperations.reduce(
            inputArray, inputOffset, IntImmutableList.of(*shape),
            outputArray, outputOffset, IntImmutableList.of(*newShape)
        )

        Assertions.assertArrayEquals(
            reducedTensor.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun multipleDimensionsCountReduceTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val tensorDimensions = source.nextInt(4, 7)
        val shape = IntArray(tensorDimensions) { source.nextInt(2, 7) }

        val newDimensionsCount = source.nextInt(tensorDimensions - 2, tensorDimensions + 1)
        val newShape = IntArray(newDimensionsCount) { shape[tensorDimensions - newDimensionsCount + it] }

        val reduceDimensionsCount = source.nextInt(1, newDimensionsCount)
        val permutation = PermutationSampler.natural(newDimensionsCount)
        PermutationSampler.shuffle(source, permutation)

        for (i in 0 until reduceDimensionsCount) {
            newShape[permutation[i]] = 1
        }

        val tensor = FloatTensor.random(source, *shape)
        val reducedTensor = tensor.reduce(*newShape)

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = FloatArray(tensor.size + inputOffset)
        tensor.toFlatArray().copyInto(inputArray, inputOffset)

        val outputArray = FloatArray(reducedTensor.size + outputOffset)

        TensorOperations.reduce(
            inputArray, inputOffset, IntImmutableList.of(*shape),
            outputArray, outputOffset, IntImmutableList.of(*newShape)
        )

        Assertions.assertArrayEquals(
            reducedTensor.toFlatArray(),
            outputArray.sliceArray(outputOffset..<outputArray.size), 0.001f
        )
    }
}