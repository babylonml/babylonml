package com.babylonml.backend.training.operations

import com.babylonml.backend.training.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.tornadoml.cpu.FloatMatrix
import com.tornadoml.cpu.SeedsArgumentsProvider
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class MultiplicationTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val firstMatrixRows = source.nextInt(100)
        val firstMatrixColumns = source.nextInt(100)

        val secondMatrixRows = firstMatrixColumns
        val secondMatrixColumns = source.nextInt(100)

        val firstMatrix = FloatMatrix.random(firstMatrixRows, firstMatrixColumns, source)
        val secondMatrix = FloatMatrix.random(secondMatrixRows, secondMatrixColumns, source)

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val firstVariable = firstMatrix.toVariable(executionContext, optimizer, learningRate)
        val secondVariable = secondMatrix.toVariable(executionContext, optimizer, learningRate)

        val multiplication = Multiplication(
            executionContext, firstMatrixRows, firstMatrixColumns, secondMatrixColumns,
            firstVariable, secondVariable
        )

        executionContext.initializeExecution(multiplication)
        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResult = firstMatrix * secondMatrix

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            buffer.copyOfRange(resultOffset, resultOffset + firstMatrixRows * secondMatrixColumns), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun differentiationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val firstMatrixRows = source.nextInt(100)
        val firstMatrixColumns = source.nextInt(100)

        val secondMatrixRows = firstMatrixColumns
        val secondMatrixColumns = source.nextInt(100)

        val firstMatrix = FloatMatrix.random(firstMatrixRows, firstMatrixColumns, source)
        val secondMatrix = FloatMatrix.random(secondMatrixRows, secondMatrixColumns, source)

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val firstVariable = firstMatrix.toVariable(executionContext, optimizer, learningRate)
        val secondVariable = secondMatrix.toVariable(executionContext, optimizer, learningRate)

        val multiplication = Multiplication(
            executionContext, firstMatrixRows, firstMatrixColumns, secondMatrixColumns,
            firstVariable, secondVariable
        )

        val gradients = FloatMatrix.random(firstMatrixRows, secondMatrixColumns, source)
        val gradientSource = GradientSource(
            executionContext, firstMatrixRows, secondMatrixColumns,
            gradients.toFlatArray(), multiplication
        )

        executionContext.initializeExecution(gradientSource)
        executionContext.executePropagation()

        val firstMatrixExpectedGradients = gradients * secondMatrix.transpose()
        val secondMatrixExpectedGradients = firstMatrix.transpose() * gradients

        val firstMatrixExpectedResult = firstMatrix - firstMatrixExpectedGradients * learningRate
        val secondMatrixExpectedResult = secondMatrix - secondMatrixExpectedGradients * learningRate

        Assertions.assertArrayEquals(
            firstMatrixExpectedResult.toFlatArray(),
            firstVariable.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            secondMatrixExpectedResult.toFlatArray(),
            secondVariable.data,
            0.001f
        )
    }
}