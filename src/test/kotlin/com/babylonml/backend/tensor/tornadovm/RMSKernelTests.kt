package com.babylonml.backend.tensor.tornadovm

import com.babylonml.AbstractTvmTest
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.backend.TvmFloatArray
import com.babylonml.backend.tensor.FloatTensor
import it.unimi.dsi.fastutil.ints.IntImmutableList
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions

import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource
import uk.ac.manchester.tornado.api.GridScheduler

class RMSKernelTests : AbstractTvmTest() {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun sqrSumTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val batchSize = source.nextInt(1, 16)
        val seqSize = source.nextInt(1, 128)
        val dim = source.nextInt(1, 128)

        val hostInput = FloatTensor.random(source, batchSize, seqSize, dim)

        val hostOutput = (hostInput * hostInput).sum(-1)
        val deviceInput = hostInput.toTvmFlatArray()

        val deviceOutput = TvmFloatArray(TvmCommons.tensorReduceResultSize(batchSize * seqSize, dim))
        val taskGraph = taskGraph(deviceInput)

        val gridScheduler = GridScheduler()
        TvmTensorOperations.addSquareSumKernel(
            taskGraph, testName, gridScheduler,
            deviceInput,
            0, batchSize * seqSize, dim, deviceOutput, 0
        )

        assertExecution(taskGraph, gridScheduler, deviceOutput) {
            Assertions.assertArrayEquals(
                hostOutput.toFlatArray(), deviceOutput.toHeapArray().copyOfRange(0, hostOutput.size),
                0.001f
            )
        }
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun f32WeightsRMSKernelTVMTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val batchSize = source.nextInt(1, 16)
        val seqSize = source.nextInt(1, 128)
        val dim = source.nextInt(1, 128)

        val input = FloatTensor.random(source, batchSize, seqSize, dim)
        val weights = FloatTensor.random(source, dim)

        val epsilon = 1e-5f
        //x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        val expectedMeanSquares = input.pow(2.0f).mean(-1, keepDim = true)
        val norm = input * (expectedMeanSquares + epsilon).rsqrt()
        //self.weight * self._norm(x.float()).type_as(x)
        val expectedRMS = norm * weights

        val inputArrayOffset = source.nextInt(8)
        val inputArray = input.toTvmFlatArray(offset = inputArrayOffset)
        val inputShape = IntImmutableList.of(*input.shape)

        val weightsArrayOffset = source.nextInt(8)
        val weightsArray = weights.toTvmFlatArray(offset = weightsArrayOffset)

        val resultArrayOffset = source.nextInt(8)
        val resultArray = TvmFloatArray(expectedRMS.size + resultArrayOffset)

        val squareSumArrayOffset = source.nextInt(8)
        val squareSumArray = TvmFloatArray(batchSize * seqSize + squareSumArrayOffset)

        val taskGraph = taskGraph(inputArray, weightsArray, squareSumArray)

        val gridScheduler = GridScheduler()
        TvmTensorOperations.addRMSNormKernel(
            taskGraph, testName, gridScheduler,
            inputArray, inputShape, inputArrayOffset, squareSumArray, squareSumArrayOffset, weightsArray,
            weightsArrayOffset, resultArray, resultArrayOffset, epsilon
        )

        assertExecution(taskGraph, gridScheduler, resultArray) {
            Assertions.assertArrayEquals(
                expectedRMS.toFlatArray(),
                resultArray.toHeapArray().copyOfRange(resultArrayOffset, resultArray.size), 0.001f
            )
        }
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun i32WeightsRMSKernelTVMTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val batchSize = source.nextInt(1, 16)
        val seqSize = source.nextInt(1, 128)
        val dim = source.nextInt(1, 128)

        val input = FloatTensor.random(source, batchSize, seqSize, dim)
        val weights = FloatTensor.randomBytes(source, dim)

        val epsilon = 1e-5f
        //x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        val expectedMeanSquares = input.pow(2.0f).mean(-1, keepDim = true)
        val norm = input * (expectedMeanSquares + epsilon).rsqrt()
        //self.weight * self._norm(x.float()).type_as(x)
        val expectedRMS = norm * weights

        val inputArrayOffset = source.nextInt(8)
        val inputArray = input.toTvmFlatArray(offset = inputArrayOffset)
        val inputShape = IntImmutableList.of(*input.shape)

        val weightsArrayOffset = source.nextInt(8)
        val weightsArray = weights.toTvmByteArray(offset = weightsArrayOffset)

        val resultArrayOffset = source.nextInt(8)
        val resultArray = TvmFloatArray(expectedRMS.size + resultArrayOffset)

        val squareSumArrayOffset = source.nextInt(8)
        val squareSumArray = TvmFloatArray(batchSize * seqSize + squareSumArrayOffset)

        val taskGraph = taskGraph(inputArray, weightsArray, squareSumArray)

        val gridScheduler = GridScheduler()
        TvmTensorOperations.addRMSNormKernel(
            taskGraph, testName, gridScheduler,
            inputArray, inputShape, inputArrayOffset, squareSumArray, squareSumArrayOffset, weightsArray,
            weightsArrayOffset, resultArray, resultArrayOffset, epsilon
        )

        assertExecution(taskGraph, gridScheduler, resultArray) {
            Assertions.assertArrayEquals(
                expectedRMS.toFlatArray(),
                resultArray.toHeapArray().copyOfRange(resultArrayOffset, resultArray.size), 0.001f
            )
        }
    }
}
