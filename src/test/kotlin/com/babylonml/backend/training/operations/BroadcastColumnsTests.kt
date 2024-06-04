package com.babylonml.backend.training.operations

import com.babylonml.backend.training.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.tornadoml.cpu.FloatMatrix
import com.tornadoml.cpu.SeedsArgumentsProvider
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class BroadcastColumnsTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(100)
        val columns = source.nextInt(100)

        val matrix = FloatMatrix.random(rows, 1, source)

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val variable = matrix.toVariable(executionContext, optimizer, learningRate)
        val broadcast = BroadcastColumns(
            rows,
            columns,
            executionContext,
            variable
        )

        executionContext.initializeExecution(broadcast)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResult = matrix.broadcastByColumns(columns)

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            buffer.copyOfRange(resultOffset, resultOffset + rows * columns), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun differentiationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(100)
        val columns = source.nextInt(100)

        val matrix = FloatMatrix.random(rows, 1, source)

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val variable = matrix.toVariable(executionContext, optimizer, learningRate)
        val broadcast = BroadcastColumns(
            rows,
            columns,
            executionContext,
            variable
        )

        val gradients = FloatMatrix.random(rows, columns, source)
        val gradientSource = GradientSource(executionContext, rows, columns, gradients.toFlatArray(), broadcast)

        executionContext.initializeExecution(gradientSource)
        executionContext.executePropagation()

        val expectedGradients = gradients.sumByColumns()
        val expectedResult = matrix - expectedGradients * learningRate

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(), variable.data, 0.001f
        )
    }
}