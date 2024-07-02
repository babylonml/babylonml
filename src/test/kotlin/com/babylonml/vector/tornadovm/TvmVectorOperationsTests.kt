package com.babylonml.vector.tornadovm

import com.babylonml.SeedsArgumentsProvider
import com.babylonml.backend.inference.operations.tornadovm.TvmFloatArray
import com.babylonml.backend.tornadovm.TvmVectorOperations
import com.babylonml.vector.FloatVector
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource
import uk.ac.manchester.tornado.api.TaskGraph
import uk.ac.manchester.tornado.api.TornadoExecutionPlan
import uk.ac.manchester.tornado.api.enums.DataTransferMode
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException

class TvmVectorOperationsTests {
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

            val firstHeapArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            firstVector.toArray().copyInto(firstHeapArray)

            val secondHeapArray = FloatArray(vectorLength + 1) {
                source.nextFloat()
            }
            secondVector.toArray().copyInto(secondHeapArray)

            val calculationResult = TvmFloatArray(vectorLength + 1)
            val firstTvmArray = TvmFloatArray.fromArray(firstHeapArray)
            val secondTvmArray = TvmFloatArray.fromArray(secondHeapArray)


            val taskGraph = TaskGraph("executionPass")
            taskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, firstHeapArray, secondHeapArray)
            taskGraph.task(
                "addVectorToVector", { first: TvmFloatArray, second: TvmFloatArray, res: TvmFloatArray ->
                    TvmVectorOperations.addVectorToVector(
                        first, 0,
                        second, 0,
                        res, 0,
                        vectorLength
                    )
                },
                firstTvmArray, secondTvmArray, calculationResult
            )
            val immutableTaskGraph = taskGraph.snapshot()
            try {
                val executionResult = TornadoExecutionPlan(immutableTaskGraph).use { executionPlan ->
                    executionPlan.execute()
                }

                executionResult.transferToHost(calculationResult)

                Assertions.assertArrayEquals(
                    (firstVector + secondVector).toArray(),
                    calculationResult.toHeapArray().copyOfRange(0, vectorLength), 0.001f
                )
            } catch (e: TornadoExecutionPlanException) {
                throw RuntimeException("Failed to execute the task graph", e)
            }
        }
    }
}