package com.babylonml.backend.inference.operations

import com.babylonml.SeedsArgumentsProvider
import com.babylonml.backend.inference.operations.tornadovm.InferenceExecutionContext
import com.babylonml.backend.inference.operations.tornadovm.TvmFloatArray
import com.babylonml.backend.tornadovm.TvmTensorOperationsTest
import com.babylonml.tensor.FloatTensor
import it.unimi.dsi.fastutil.ints.IntImmutableList
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class RoPEOperationTest {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testRotationGeneration(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val bs = source.nextInt(1, 16)
        val seqLen = (source.nextInt(2, 65) shr 1) shl 1
        val numHeads = source.nextInt(1, 16)
        val headDim = (source.nextInt(2, 65) shr 1) shl 1
        val queryTensor = FloatTensor.random(source, bs, seqLen, numHeads, headDim)

        InferenceExecutionContext().use {
            val query = InputSourceOperation(
                queryTensor.toFlatArray(),
                IntImmutableList.of(*queryTensor.shape), "query", it
            )
            val startPosition = InputSourceOperation(
                floatArrayOf(0.0f), IntImmutableList.of(1), "startPosition", it
            )

            val (expectedSin, expectedCos) = TvmTensorOperationsTest.prepareRotationTensors(headDim, seqLen)

            val roPEOperation = RoPEOperation("rope", query, startPosition)
            val actualCos = TvmFloatArray(expectedCos.size)

            val invFreqs = roPEOperation.generateInvFreqs(headDim)
            roPEOperation.fillCosTensor(seqLen, headDim, invFreqs, actualCos, 0)

            val actualSin = TvmFloatArray(expectedSin.size)
            roPEOperation.fillSinTensor(seqLen, headDim, invFreqs, actualSin, 0)

            Assertions.assertArrayEquals(expectedCos.toFlatArray(), actualCos.toHeapArray(), 0.0001f)
            Assertions.assertArrayEquals(expectedSin.toFlatArray(), actualSin.toHeapArray(), 0.0001f)
        }
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testRoPESingleExecutionFromStart(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val bs = source.nextInt(1, 16)
        val seqLen = (source.nextInt(2, 65) shr 1) shl 1
        val numHeads = source.nextInt(1, 16)
        val headDim = (source.nextInt(2, 65) shr 1) shl 1
        val queryTensor = FloatTensor.random(source, bs, seqLen, numHeads, headDim)

        val (expectedSin, expectedCos) = TvmTensorOperationsTest.prepareRotationTensors(headDim, seqLen)
        val expectedResult = TvmTensorOperationsTest.applyRotation(queryTensor, expectedCos, expectedSin)

        InferenceExecutionContext().use {
            val query = InputSourceOperation(
                queryTensor.toFlatArray(),
                IntImmutableList.of(*queryTensor.shape), "query", it
            )
            val startPosition = InputSourceOperation(
                floatArrayOf(0.0f), IntImmutableList.of(1), "startPosition", it
            )
            val roPEOperation = RoPEOperation("rope", query, startPosition)

            it.initializeExecution(roPEOperation)
            val result = it.executePass()

            Assertions.assertArrayEquals(expectedResult.toFlatArray(), result, 0.0001f)
        }
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testRoPEMultiExecutionFromStart(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val bs = source.nextInt(1, 16)
        val seqLen = (source.nextInt(2, 65) shr 1) shl 1
        val numHeads = source.nextInt(1, 16)
        val headDim = (source.nextInt(2, 65) shr 1) shl 1

        var queryTensor = FloatTensor.random(source, bs, seqLen, numHeads, headDim)

        val (expectedSin, expectedCos) = TvmTensorOperationsTest.prepareRotationTensors(headDim, seqLen)
        var expectedResult = TvmTensorOperationsTest.applyRotation(queryTensor, expectedCos, expectedSin)

        InferenceExecutionContext().use {
            val query = InputSourceOperation(
                queryTensor.toFlatArray(),
                IntImmutableList.of(*queryTensor.shape), "query", it
            )
            val startPosition = InputSourceOperation(
                floatArrayOf(0.0f), IntImmutableList.of(1), "startPosition", it
            )
            val roPEOperation = RoPEOperation("rope", query, startPosition)

            it.initializeExecution(roPEOperation)
            val iterationsCount = source.nextInt(1, 8)
            for (iteration in 0 until iterationsCount) {
                val result = it.executePass()

                Assertions.assertArrayEquals(expectedResult.toFlatArray(), result, 0.0001f)

                queryTensor = FloatTensor.random(source, bs, seqLen, numHeads, headDim)
                expectedResult = TvmTensorOperationsTest.applyRotation(queryTensor, expectedCos, expectedSin)

                query.value = queryTensor.toFlatArray()
            }
        }
    }
}