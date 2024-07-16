package com.babylonml.backend.tornadovm

import com.babylonml.AbstractTvmTest
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.backend.inference.operations.tornadovm.TvmFloatArray
import com.babylonml.backend.inference.operations.tornadovm.TvmIntArray
import com.babylonml.tensor.FloatTensor
import it.unimi.dsi.fastutil.ints.IntImmutableList
import org.apache.commons.rng.sampling.PermutationSampler
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource
import com.babylonml.tensor.div
import com.babylonml.tensor.pow
import com.babylonml.tensor.times


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
        val shape =
            IntArray(tensorDimensions) { newShape[newTensorDimensions - tensorDimensions + it] }

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

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun ropeKernelSeqFromStartTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val bs = source.nextInt(1, 16)
        val seqLen = (source.nextInt(2, 65) shr 1) shl 1
        val numHeads = source.nextInt(1, 16)
        val headDim = (source.nextInt(2, 65) shr 1) shl 1

        val input = FloatTensor.random(source, bs, seqLen, numHeads, headDim)

        val (sin, cos) = prepareRotationTensors(headDim, seqLen)
        val expectedResult = applyRotation(input, cos, sin)

        val inputOffset = source.nextInt(8)
        val cosOffset = source.nextInt(8)
        val sinOffset = source.nextInt(8)
        val startPositionOffset = source.nextInt(8)
        val resultOffset = source.nextInt(8)


        val inputArray = input.toTvmFlatArray(offset = inputOffset)
        val resultArray = TvmFloatArray(bs * seqLen * numHeads * headDim + resultOffset)

        val startPositionArray = TvmFloatArray(1 + startPositionOffset)
        startPositionArray.set(startPositionOffset, 0.0f)

        val inputShape = TvmIntArray.fromArray(input.shape)
        val cosArray = cos.toTvmFlatArray(offset = cosOffset)
        val sinArray = sin.toTvmFlatArray(offset = sinOffset)


        TvmTensorOperations.ropeKernel(
            inputArray, inputShape, inputOffset, cosArray, cosOffset, sinArray, sinOffset,
            startPositionArray, startPositionOffset, resultArray, resultOffset, seqLen
        )

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            resultArray.toHeapArray().copyOfRange(resultOffset, resultArray.size), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun ropeTVMSeqFromStartTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val bs = source.nextInt(1, 16)
        val seqLen = (source.nextInt(2, 65) shr 1) shl 1
        val numHeads = source.nextInt(1, 16)
        val headDim = (source.nextInt(2, 65) shr 1) shl 1
        val input = FloatTensor.random(source, bs, seqLen, numHeads, headDim)

        val (sin, cos) = prepareRotationTensors(headDim, seqLen)
        val expectedResult = applyRotation(input, cos, sin)

        val inputArray = input.toTvmFlatArray()
        val resultArray = TvmFloatArray(bs * seqLen * numHeads * headDim)
        val startPositionArray = TvmFloatArray(1)
        startPositionArray.set(0, 0.0f)

        val inputShape = IntImmutableList.of(*input.shape)
        val cosArray = cos.toTvmFlatArray()
        val sinArray = sin.toTvmFlatArray()

        val taskGraph = taskGraph(inputArray, startPositionArray, cosArray, sinArray)
        TvmTensorOperations.addRopeKernel(
            taskGraph, "ropeTestSeqFromStart", inputArray,
            inputShape, 0,
            cosArray, 0,
            sinArray, 0,
            startPositionArray, 0,
            resultArray, inputShape, 0,
            seqLen
        )

        assertExecution(taskGraph, resultArray) {
            Assertions.assertArrayEquals(
                expectedResult.toFlatArray(),
                resultArray.toHeapArray(), 0.001f
            )
        }
    }

    companion object {
        private fun rotateInput(
            input: FloatTensor,
        ): FloatTensor {
            val headDim = input.shape[3]
            val input1 = input.slice(
                0 until headDim / 2
            )
            val input2 = input.slice(
                headDim / 2 until headDim
            )

            return (-1 * input2).cat(input1)
        }

        fun prepareRotationTensors(
            headDim: Int,
            seqLen: Int
        ): Pair<FloatTensor, FloatTensor> {
            // inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            val invFreqs = 1.0f / 10_000.0f.pow(
                FloatTensor.arrange(0, headDim, 2) / headDim
            )

            Assertions.assertEquals(invFreqs.size, headDim / 2)

            //t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
            // freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            val freqs = FloatTensor.arrange(end = seqLen).combineWith(invFreqs) { a, b -> a * b }

            Assertions.assertArrayEquals(freqs.shape, intArrayOf(seqLen, headDim / 2))

            val sin = freqs.sin()
            val cos = freqs.cos()

            return Pair(sin, cos)
        }

        fun applyRotation(
            input: FloatTensor,
            cos: FloatTensor,
            sin: FloatTensor
        ): FloatTensor {
            val broadcastSin = sin.cat(sin).unsquize(0).unsquize(2)
            val broadcastCos = cos.cat(cos).unsquize(0).unsquize(2)

            val rotatedInput = rotateInput(input)
            Assertions.assertArrayEquals(rotatedInput.shape, input.shape)

            return input * broadcastCos + rotatedInput * broadcastSin
        }
    }
}