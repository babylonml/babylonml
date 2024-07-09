package com.babylonml.backend.training.operations

import com.babylonml.backend.training.optimizer.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import com.babylonml.matrix.FloatMatrix
import com.babylonml.SeedsArgumentsProvider
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class MultiplicationTests {
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val firstMatrixRows = source.nextInt(1, 100)
        val firstMatrixColumns = source.nextInt(1, 100)

        val secondMatrixRows = firstMatrixColumns
        val secondMatrixColumns = source.nextInt(1, 100)

        val inputMatrix = FloatMatrix.random(firstMatrixRows, firstMatrixColumns, source)
        val variableMatrix = FloatMatrix.random(secondMatrixRows, secondMatrixColumns, source)

        val executionContext = TrainingExecutionContext(1, firstMatrixRows)
        val inputSource = executionContext.registerMainInputSource(inputMatrix.toTensor())

        val optimizer = SimpleGradientDescentOptimizer(inputSource)
        val learningRate = 0.01f

        val variable = variableMatrix.toVariable(executionContext, optimizer, learningRate)
        val multiplication = Multiplication(inputSource, variable)

        val resultCell = ResultMemoryCellCostFunction(multiplication)
        executionContext.initializeExecution(resultCell)
        executionContext.executePropagation()

        val expectedResult = inputMatrix * variableMatrix

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            resultCell.result, 0.001f
        )
    }

    fun differentiationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val firstMatrixRows = source.nextInt(1, 100)
        val firstMatrixColumns = source.nextInt(1, 100)

        val secondMatrixRows = firstMatrixColumns
        val secondMatrixColumns = source.nextInt(1, 100)

        val firstMatrix = FloatMatrix.random(firstMatrixRows, firstMatrixColumns, source)
        val secondMatrix = FloatMatrix.random(secondMatrixRows, secondMatrixColumns, source)
        val inputMatrix = FloatMatrix(firstMatrixRows, firstMatrixColumns)

        val executionContext = TrainingExecutionContext(1)
        val inputSource = executionContext.registerMainInputSource(inputMatrix.toTensor(3))
        val optimizer = SimpleGradientDescentOptimizer(inputSource)
        val learningRate = 0.01f

        val firstVariable = firstMatrix.toVariable("first", executionContext, optimizer, learningRate)
        val secondVariable = secondMatrix.toVariable("second", executionContext, optimizer, learningRate)

        val add = Add(inputSource, firstVariable)
        val multiplication = Multiplication(add, secondVariable)

        val gradients = FloatMatrix.random(firstMatrixRows, secondMatrixColumns, source)
        val gradientSource = GradientSource(
            gradients.toTensor(3), multiplication
        )

        val firstMatrixExpectedGradients = gradients * secondMatrix.transpose()
        val secondMatrixExpectedGradients = firstMatrix.transpose() * gradients

        val firstMatrixExpectedResult = firstMatrix - firstMatrixExpectedGradients * learningRate
        val secondMatrixExpectedResult = secondMatrix - secondMatrixExpectedGradients * learningRate

        executionContext.initializeExecution(gradientSource)
        executionContext.executePropagation()

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