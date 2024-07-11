package com.babylonml.backend.inference.operations


import com.babylonml.SeedsArgumentsProvider
import com.babylonml.backend.inference.operations.tornadovm.InferenceExecutionContext
import com.babylonml.tensor.ByteTensor
import com.babylonml.tensor.FloatTensor
import it.unimi.dsi.fastutil.ints.IntImmutableList
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class AddOperationTest {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun addByteToFloatTestNoBroadcastSingleExecution(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val weightsTensor = ByteTensor.random(source, rows, columns)
        val dataTensor = FloatTensor.random(source, rows, columns)

        val executionContext = InferenceExecutionContext()
        val weights = WeightsOperation(
            "weights", executionContext, weightsTensor.toFlatArray(),
            IntImmutableList.of(*weightsTensor.shape)
        )
        val data = InputSourceOperation(
            dataTensor.toFlatArray(),
            IntImmutableList.of(*dataTensor.shape), "data", executionContext
        )
        val add = AddOperation("add", weights, data)

        executionContext.initializeExecution(add)
        val result = executionContext.executePass()
        val expectedResult = weightsTensor + dataTensor

        Assertions.assertArrayEquals(expectedResult.toFlatArray(), result)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun addByteToFloatTestNoBroadcastMultiExecution(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val weightsTensor = ByteTensor.random(source, rows, columns)
        var dataTensor = FloatTensor.random(source, rows, columns)

        val executionContext = InferenceExecutionContext()
        val weights = WeightsOperation(
            "weights", executionContext, weightsTensor.toFlatArray(),
            IntImmutableList.of(*weightsTensor.shape)
        )
        val data = InputSourceOperation(
            dataTensor.toFlatArray(),
            IntImmutableList.of(*dataTensor.shape), "data", executionContext
        )
        val add = AddOperation("add", weights, data)

        executionContext.initializeExecution(add)

        var result = executionContext.executePass()
        var expectedResult = weightsTensor + dataTensor

        Assertions.assertArrayEquals(expectedResult.toFlatArray(), result)

        val iterations = source.nextInt(1, 5)
        for (i in 0 until iterations) {
            dataTensor = FloatTensor.random(source, rows, columns)
            data.value = dataTensor.toFlatArray()

            result = executionContext.executePass()
            expectedResult = weightsTensor + dataTensor

            Assertions.assertArrayEquals(expectedResult.toFlatArray(), result)
        }
    }
}