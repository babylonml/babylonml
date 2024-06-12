package com.babylonml.backend.training.operations

import com.babylonml.backend.training.optimizer.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.babylonml.matrix.FloatMatrix
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.mseCostFunctionByRows
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
        val optimizer =
            SimpleGradientDescentOptimizer(NullDataSource())
        val learningRate = 0.01f

        val predictedValuesVariable = predictedValuesMatrix.toVariable(
            executionContext, optimizer, learningRate
        )
        val expectedValuesConst = Constant(
            executionContext, expectedValuesMatrix.toFlatArray(), rows, columns
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, predictedValuesVariable, expectedValuesConst
        )

        executionContext.initializeExecution(mseCostFunction)
        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

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
        val optimizer =
            SimpleGradientDescentOptimizer(NullDataSource())
        val learningRate = 0.01f

        val predictedValuesVariable = predictedValuesMatrix.toVariable(
            executionContext, optimizer, learningRate
        )
        val expectedValuesConst = Constant(
            executionContext, expectedValuesMatrix.toFlatArray(), rows, columns
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, predictedValuesVariable, expectedValuesConst
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