package com.babylonml.backend.training.operations

import com.babylonml.backend.training.optimizer.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.babylonml.matrix.FloatMatrix
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.crossEntropyByRows
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class SoftMaxByRowsCrossEntropyTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val matrix = FloatMatrix.random(rows, columns, source)
        val expectedProbabilitiesMatrix = FloatMatrix.random(
            rows, columns,
            0f, 1f, source
        )

        val executionContext = TrainingExecutionContext()
        val optimizer =
            SimpleGradientDescentOptimizer(NullDataSource())
        val learningRate = 0.01f

        val variable = matrix.toVariable(executionContext, optimizer, learningRate)
        val softMax = SoftMaxByRows(
            executionContext,
            variable,
        )
        val crossEntropy = CrossEntropyByRowsFunction(
            expectedProbabilitiesMatrix.toConstant(executionContext),
            executionContext, softMax
        )

        executionContext.initializeExecution(crossEntropy)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResult = crossEntropyByRows(
            matrix.softMaxByRows(),
            expectedProbabilitiesMatrix
        )

        Assertions.assertEquals(expectedResult, buffer[resultOffset], 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun differentiationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val matrix = FloatMatrix.random(rows, columns, source)
        val expectedMatrix = FloatMatrix.random(
            rows, columns,
            0f, 1f, source
        )

        val executionContext = TrainingExecutionContext()
        val optimizer =
            SimpleGradientDescentOptimizer(NullDataSource())
        val learningRate = 0.01f

        val variable = matrix.toVariable(executionContext, optimizer, learningRate)
        val softMax = SoftMaxByRows(
            executionContext,
            variable
        )
        val crossEntropy = CrossEntropyByRowsFunction(
            expectedMatrix.toConstant(executionContext),
            executionContext, softMax
        )

        executionContext.initializeExecution(crossEntropy)
        executionContext.executePropagation(1)

        val expectedGradients = matrix.softMaxByRows() - expectedMatrix

        val expectedResult = matrix - expectedGradients * learningRate
        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            variable.data,
            0.001f
        )
    }
}