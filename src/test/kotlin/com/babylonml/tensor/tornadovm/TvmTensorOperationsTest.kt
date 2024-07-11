package com.babylonml.tensor.tornadovm

import com.babylonml.AbstractTvmTest
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.backend.inference.operations.tornadovm.TvmFloatArray
import com.babylonml.backend.tornadovm.TvmTensorOperations
import com.babylonml.tensor.FloatTensor
import it.unimi.dsi.fastutil.ints.IntImmutableList
import org.apache.commons.rng.sampling.PermutationSampler
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource


class TvmTensorOperationsTest : AbstractTvmTest() {
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

        val inputArray = vector.toTvmFlatArray(offset = inputOffset)
        val outputArray = TvmFloatArray(matrixRows * vectorSize + outputOffset)

        val taskGraph = taskGraph(inputArray)

        TvmTensorOperations.addBroadcastTask(
            taskGraph, "vectorToMatrixByRowsBroadcastTest",
            inputArray, inputOffset, IntImmutableList.of(vectorSize),
            outputArray, outputOffset, IntImmutableList.of(matrixRows, vectorSize)
        )
        assertExecution(taskGraph, outputArray) {
            Assertions.assertArrayEquals(
                matrix.toFlatArray(),
                outputArray.toHeapArray().sliceArray(outputOffset..<outputArray.size), 0.001f
            )
        }
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun matrixToMatrixByColumnsBroadcastTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val matrixRows = source.nextInt(1, 100)
        val initialMatrix = FloatTensor.natural(matrixRows, 1)

        val matrixColumns = source.nextInt(1, 100)
        val expectedResultMatrix = initialMatrix.broadcast(matrixRows, matrixColumns)

        val inputOffset = source.nextInt(8)
        val outputOffset = source.nextInt(8)

        val inputArray = initialMatrix.toTvmFlatArray(offset = inputOffset)
        val outputArray = TvmFloatArray(expectedResultMatrix.size + outputOffset)

        val taskGraph = taskGraph(inputArray)

        TvmTensorOperations.addBroadcastTask(
            taskGraph, "matrixToMatrixByColumnsBroadcastTest",
            inputArray, inputOffset, IntImmutableList.of(matrixRows, 1),
            outputArray, outputOffset, IntImmutableList.of(matrixRows, matrixColumns)
        )

        assertExecution(taskGraph, outputArray) {
            Assertions.assertArrayEquals(
                expectedResultMatrix.toFlatArray(),
                outputArray.toHeapArray().sliceArray(outputOffset..<outputArray.size),
                0.001f
            )
        }
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

        val inputArray = initialMatrix.toTvmFlatArray(offset = inputOffset)

        val outputArray = TvmFloatArray(matrixColumns * matrixRows + outputOffset)

        val taskGraph = taskGraph(inputArray)

        TvmTensorOperations.addBroadcastTask(
            taskGraph, "scalarToMatrixBroadcastTest",
            inputArray, inputOffset, IntImmutableList.of(1, 1),
            outputArray, outputOffset, IntImmutableList.of(matrixRows, matrixColumns)
        )

        assertExecution(taskGraph, outputArray) {
            Assertions.assertArrayEquals(
                expectedResultMatrix.toFlatArray(),
                outputArray.toHeapArray().sliceArray(outputOffset..<outputArray.size), 0.001f
            )
        }
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

        val inputArray = tensor.toTvmFlatArray(offset = inputOffset)

        val outputArray = TvmFloatArray(broadcastTensor.size + outputOffset)

        val taskGraph = taskGraph(inputArray)
        TvmTensorOperations.addBroadcastTask(
            taskGraph, "multipleDimensionsBroadcastTest",
            inputArray, inputOffset, IntImmutableList.of(*shape),
            outputArray, outputOffset, IntImmutableList.of(*newShape)
        )


        assertExecution(taskGraph, outputArray) {
            Assertions.assertArrayEquals(
                broadcastTensor.toFlatArray(),
                outputArray.toHeapArray().sliceArray(outputOffset..<outputArray.size), 0.001f
            )
        }
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

        val inputArray = tensor.toTvmFlatArray(offset = inputOffset)
        val outputArray = TvmFloatArray(broadcastTensor.size + outputOffset)

        val taskGraph = taskGraph(inputArray)

        TvmTensorOperations.addBroadcastTask(
            taskGraph, "multipleDimensionsCountBroadcastTest",
            inputArray, inputOffset, IntImmutableList.of(*shape),
            outputArray, outputOffset, IntImmutableList.of(*newShape)
        )

        assertExecution(taskGraph, outputArray) {
            Assertions.assertArrayEquals(
                broadcastTensor.toFlatArray(),
                outputArray.toHeapArray().sliceArray(outputOffset..<outputArray.size), 0.001f
            )
        }
    }
}