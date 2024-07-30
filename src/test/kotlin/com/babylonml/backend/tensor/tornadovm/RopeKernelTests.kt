package com.babylonml.backend.tensor.tornadovm

import com.babylonml.AbstractTvmTest
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.backend.TvmFloatArray
import com.babylonml.backend.TvmIntArray
import com.babylonml.backend.tensor.FloatTensor
import it.unimi.dsi.fastutil.ints.IntImmutableList
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource
import com.babylonml.backend.tensor.div
import com.babylonml.backend.tensor.pow
import com.babylonml.backend.tensor.times
import uk.ac.manchester.tornado.api.GridScheduler


class RopeKernelTests : AbstractTvmTest() {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun ropeSeqFromStartTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val bs = source.nextInt(1, 16)
        val seqLen = (source.nextInt(2, 65) shr 1) shl 1
        val numHeads = source.nextInt(1, 16)
        val headDim = (source.nextInt(2, 65) shr 1) shl 1
        val input = FloatTensor.random(source, bs, seqLen, numHeads, headDim)

        val (sin, cos) = prepareRotationTensors(headDim, seqLen)
        val expectedResult = applyRotation(input, cos, sin, 0)

        val inputOffset = source.nextInt(8)
        val cosOffset = source.nextInt(8)
        val sinOffset = source.nextInt(8)
        val startPositionOffset = source.nextInt(8)
        val resultOffset = source.nextInt(8)

        val inputArray = input.toTvmFlatArray(offset = inputOffset)
        val resultArray = TvmFloatArray(bs * seqLen * numHeads * headDim + resultOffset)
        val startPositionArray = TvmIntArray(1 + startPositionOffset)
        startPositionArray.set(startPositionOffset, 0)

        val inputShape = IntImmutableList.of(*input.shape)
        val cosArray = cos.toTvmFlatArray(offset = cosOffset)
        val sinArray = sin.toTvmFlatArray(offset = sinOffset)

        val taskGraph = taskGraph(inputArray, startPositionArray, cosArray, sinArray)
        val gridScheduler = GridScheduler()
        TvmTensorOperations.addRopeKernel(
            taskGraph, "ropeTVMSeqFromStartTest", gridScheduler, inputArray,
            inputShape, inputOffset,
            cosArray, cosOffset,
            sinArray, sinOffset,
            startPositionArray, startPositionOffset,
            resultArray, inputShape, resultOffset,
            seqLen
        )

        assertExecution(taskGraph, gridScheduler, resultArray) {
            Assertions.assertArrayEquals(
                expectedResult.toFlatArray(),
                resultArray.toHeapArray().copyOfRange(resultOffset, resultArray.size), 0.001f
            )
        }
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun ropeSeqSliceTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val bs = source.nextInt(1, 16)
        val seqLen = (source.nextInt(2, 16) shr 1) shl 1
        val numHeads = source.nextInt(1, 16)
        val headDim = (source.nextInt(2, 65) shr 1) shl 1
        val maxSeqLen = (source.nextInt(seqLen, 2 * seqLen) shr 1) shl 1

        val input = FloatTensor.random(source, bs, seqLen, numHeads, headDim)

        val (sin, cos) = prepareRotationTensors(headDim, maxSeqLen)

        val startPosition = source.nextInt(0, maxSeqLen - seqLen + 1)
        val expectedResult = applyRotation(input, cos, sin, startPosition)

        val inputOffset = source.nextInt(8)
        val cosOffset = source.nextInt(8)
        val sinOffset = source.nextInt(8)
        val startPositionOffset = source.nextInt(8)
        val resultOffset = source.nextInt(8)

        val inputArray = input.toTvmFlatArray(offset = inputOffset)
        val resultArray = TvmFloatArray(bs * seqLen * numHeads * headDim + resultOffset)

        val startPositionArray = TvmIntArray(1 + startPositionOffset)

        startPositionArray.set(startPositionOffset, startPosition)

        val inputShape = IntImmutableList.of(*input.shape)
        val cosArray = cos.toTvmFlatArray(offset = cosOffset)
        val sinArray = sin.toTvmFlatArray(offset = sinOffset)

        val taskGraph = taskGraph(inputArray, startPositionArray, cosArray, sinArray)

        val gridScheduler = GridScheduler()
        TvmTensorOperations.addRopeKernel(
            taskGraph, "ropeTVMSeqSliceTest", gridScheduler, inputArray,
            inputShape, inputOffset,
            cosArray, cosOffset,
            sinArray, sinOffset,
            startPositionArray, startPositionOffset,
            resultArray, inputShape, resultOffset,
            seqLen
        )

        assertExecution(taskGraph, gridScheduler, resultArray) {
            Assertions.assertArrayEquals(
                expectedResult.toFlatArray(),
                resultArray.toHeapArray().copyOfRange(resultOffset, resultArray.size), 0.001f
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
            sin: FloatTensor,
            startPosition: Int
        ): FloatTensor {
            val rotationShape = sin.shape
            Assertions.assertArrayEquals(cos.shape, rotationShape)

            val inputShape = input.shape

            val inputSeqLen = inputShape[1]
            val headDim = inputShape[3]

            val doubleSin = sin.cat(sin)
            val doubleCos = cos.cat(cos)

            val doubleSinSlice = doubleSin.slice(startPosition until startPosition + inputSeqLen, 0 until headDim)
            val doubleCosSlice = doubleCos.slice(startPosition until startPosition + inputSeqLen, 0 until headDim)

            val broadcastSin = doubleSinSlice.unsquize(0).unsquize(2)
            val broadcastCos = doubleCosSlice.unsquize(0).unsquize(2)

            val rotatedInput = rotateInput(input)
            Assertions.assertArrayEquals(rotatedInput.shape, input.shape)

            return input * broadcastCos + rotatedInput * broadcastSin
        }
    }
}