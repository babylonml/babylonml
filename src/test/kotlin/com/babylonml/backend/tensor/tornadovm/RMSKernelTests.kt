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
    fun singleTensorSqrSumTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val batchSize = source.nextInt(1, 16)
        val seqSize = source.nextInt(1, 128)
        val dim = source.nextInt(1, 128)

        val hostInput = FloatTensor.random(source, batchSize, seqSize, dim)

        val hostOutput = (hostInput * hostInput).sum(-1)
        val inputShape = IntImmutableList.of(*hostInput.shape)

        val deviceInput = hostInput.toTvmFlatArray()

        val deviceOutput = TvmFloatArray(TvmTensorOperations.tensorReduceResultSize(inputShape))
        val taskGraph = taskGraph(deviceInput)

        val gridScheduler = GridScheduler()
        TvmTensorOperations.addSquareSumKernel(
            taskGraph, TvmCommons.generateName("singleTensorSqrSumTest"), gridScheduler,
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
}
