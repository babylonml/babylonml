package com.babylonml.backend.training.operations

import com.babylonml.backend.training.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.tornadoml.cpu.FloatMatrix
import com.tornadoml.cpu.SeedsArgumentsProvider
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
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val firstVariable = firstMatrix.toVariable(executionContext, optimizer, learningRate)
        val secondVariable = secondMatrix.toVariable(executionContext, optimizer, learningRate)

        val add = Add(executionContext, rows, columns, firstVariable, secondVariable)

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
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val firstVariable = firstMatrix.toVariable(executionContext, optimizer, learningRate)
        val secondVariable = secondMatrix.toVariable(executionContext, optimizer, learningRate)

        val add = Add(executionContext, rows, columns, firstVariable, secondVariable)
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