package com.babylonml.backend.training.operations

import com.babylonml.backend.training.optimizer.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.babylonml.matrix.FloatMatrix
import com.babylonml.SeedsArgumentsProvider
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class AddTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(100)
        val columns = source.nextInt(100)

        val firstMatrix = FloatMatrix.random(rows, columns, source)
        val secondMatrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext()
        val optimizer =
            SimpleGradientDescentOptimizer(NullDataSource())
        val learningRate = 0.01f

        val firstVariable = firstMatrix.toVariable(executionContext, optimizer, learningRate)
        val secondVariable = secondMatrix.toVariable(executionContext, optimizer, learningRate)

        val add = Add(executionContext, firstVariable, secondVariable, false)

        executionContext.initializeExecution(add)
        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResult = firstMatrix + secondMatrix

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

        val firstMatrix = FloatMatrix.random(rows, columns, source)
        val secondMatrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext()
        val optimizer =
            SimpleGradientDescentOptimizer(NullDataSource())
        val learningRate = 0.01f

        val firstVariable = firstMatrix.toVariable(executionContext, optimizer, learningRate)
        val secondVariable = secondMatrix.toVariable(executionContext, optimizer, learningRate)

        val add = Add(executionContext, firstVariable, secondVariable, false)
        val gradients = FloatMatrix.random(rows, columns, source)

        val gradientSource = GradientSource(executionContext, rows, columns, gradients.toFlatArray(), add)

        executionContext.initializeExecution(gradientSource)
        executionContext.executePropagation(1)

        val firstExpectedResult = firstMatrix - gradients * learningRate
        val secondExpectedResult = secondMatrix - gradients * learningRate

        Assertions.assertArrayEquals(firstExpectedResult.toFlatArray(), firstVariable.data, 0.001f)
        Assertions.assertArrayEquals(secondExpectedResult.toFlatArray(), secondVariable.data, 0.001f)
    }
}