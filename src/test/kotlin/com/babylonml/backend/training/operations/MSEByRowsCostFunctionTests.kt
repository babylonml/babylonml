package com.babylonml.backend.training.operations

import com.babylonml.backend.training.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.tornadoml.cpu.FloatMatrix
import com.tornadoml.cpu.SeedsArgumentsProvider
import com.tornadoml.cpu.mseCostFunctionByRows
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class MSEByRowsCostFunctionTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val predictedValuesMatrix = FloatMatrix.random(rows, columns, source)
        val expectedValuesMatrix = FloatMatrix.random(
            rows, columns, source
        )

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val predictedValuesVariable = predictedValuesMatrix.toVariable(
            executionContext, optimizer, learningRate
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, predictedValuesVariable, rows, columns, expectedValuesMatrix.toFlatArray()
        )

        executionContext.initializeExecution(mseCostFunction)
        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        Assertions.assertEquals(1, TrainingExecutionContext.addressLength(result))
        val expectedResult = mseCostFunctionByRows(
            predictedValuesMatrix, expectedValuesMatrix
        )

        Assertions.assertEquals(expectedResult, buffer[resultOffset], 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun differentiationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val predictedValuesMatrix = FloatMatrix.random(rows, columns, source)
        val expectedValuesMatrix = FloatMatrix.random(
            rows, columns, source
        )

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val predictedValuesVariable = predictedValuesMatrix.toVariable(
            executionContext, optimizer, learningRate
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, predictedValuesVariable, rows, columns, expectedValuesMatrix.toFlatArray()
        )

        executionContext.initializeExecution(mseCostFunction)
        executionContext.executePropagation(1)

        val expectedGradients = predictedValuesMatrix - expectedValuesMatrix


        val expectedResult = predictedValuesMatrix - expectedGradients * learningRate
        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            predictedValuesVariable.data,
            0.001f
        )
    }
}