package com.babylonml.vector.tornadovm

import com.babylonml.AbstractTvmTest
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.backend.inference.operations.tornadovm.TvmFloatArray
import com.babylonml.backend.tornadovm.TvmVectorOperations
import com.babylonml.vector.FloatVector
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource
import uk.ac.manchester.tornado.api.enums.DataTransferMode


class TvmVectorOperationsTests : AbstractTvmTest() {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun addVectorToVectorTest(seed: Long) {
        for (n in 0..9) {
            val source = RandomSource.ISAAC.create(seed)

            val vectorLength = source.nextInt(1000)

            val firstVector = FloatVector(vectorLength)
            val secondVector = FloatVector(vectorLength)

            firstVector.fillRandom(source)
            secondVector.fillRandom(source)

            val calculationResult = TvmFloatArray(vectorLength + 1)
            val firstTvmArray = firstVector.toTvmFloatArray(length = vectorLength + 1)
            val secondTvmArray = secondVector.toTvmFloatArray(length = vectorLength + 1)

            val taskGraph = taskGraph(firstTvmArray, secondTvmArray)
            TvmVectorOperations.addVectorToVectorTask(
                taskGraph, "addVectorToVector",
                firstTvmArray, 0, secondTvmArray, 0, calculationResult, 0, vectorLength
            )
            taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, calculationResult)
            assertExecution(taskGraph) {
                Assertions.assertArrayEquals(
                    (firstVector + secondVector).toArray(),
                    calculationResult.toHeapArray().copyOfRange(0, vectorLength), 0.001f
                )
            }
        }
    }
}