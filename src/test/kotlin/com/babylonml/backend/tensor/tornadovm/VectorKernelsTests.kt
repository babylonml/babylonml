package com.babylonml.backend.tensor.tornadovm

import com.babylonml.AbstractTvmTest
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.backend.TvmFloatArray
import com.babylonml.backend.tensor.FloatTensor
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource
import uk.ac.manchester.tornado.api.GridScheduler


class VectorKernelsTests : AbstractTvmTest() {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun addVectorToVectorTest(seed: Long) {
        for (n in 0..9) {
            val source = RandomSource.ISAAC.create(seed)

            val vectorLength = source.nextInt(1000)

            val firstVector = FloatTensor.random(source, vectorLength)
            val secondVector = FloatTensor.random(source, vectorLength)

            val resultOffset = source.nextInt(8)
            val firstVectorOffset = source.nextInt(8)
            val secondVectorOffset = source.nextInt(8)

            val calculationResult = TvmFloatArray(vectorLength + resultOffset)
            val firstTvmArray = firstVector.toTvmFlatArray(offset = firstVectorOffset)
            val secondTvmArray = secondVector.toTvmFlatArray(offset = secondVectorOffset)

            val taskGraph = taskGraph(firstTvmArray, secondTvmArray)
            val gridScheduler = GridScheduler()

            TvmVectorOperations.addVectorToVectorTask(
                taskGraph, "addVectorToVector", gridScheduler,
                firstTvmArray, firstVectorOffset, secondTvmArray, secondVectorOffset,
                calculationResult, resultOffset, vectorLength
            )

            assertExecution(taskGraph, gridScheduler, calculationResult) {
                Assertions.assertArrayEquals(
                    (firstVector + secondVector).toFlatArray(),
                    calculationResult.toHeapArray()
                        .copyOfRange(resultOffset, calculationResult.size), 0.001f
                )
            }
        }
    }
}