package com.babylonml.backend.training.operations

import com.babylonml.backend.training.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.tornadoml.cpu.FloatMatrix
import com.tornadoml.cpu.SeedsArgumentsProvider
import com.tornadoml.cpu.crossEntropyByRows
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
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val variable = matrix.toVariable(executionContext, optimizer, learningRate)
        val softMax = SoftMaxByRows(
            executionContext,
            variable,
            rows,
            columns
        )
        val crossEntropy = CrossEntropyByRowsFunction(
            rows, columns, expectedProbabilitiesMatrix.toFlatArray(),
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
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val variable = matrix.toVariable(executionContext, optimizer, learningRate)
        val softMax = SoftMaxByRows(
            executionContext,
            variable,
            rows,
            columns
        )
        val crossEntropy = CrossEntropyByRowsFunction(
            rows, columns, expectedMatrix.toFlatArray(),
            executionContext, softMax
        )

        executionContext.initializeExecution(crossEntropy)
        executionContext.executePropagation()

        val expectedGradients = matrix.softMaxByRows() - expectedMatrix

        val expectedResult = matrix - expectedGradients * learningRate
        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            variable.data,
            0.001f
        )
    }
}