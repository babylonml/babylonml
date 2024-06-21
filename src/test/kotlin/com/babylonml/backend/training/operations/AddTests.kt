package com.babylonml.backend.training.operations

import com.babylonml.backend.training.optimizer.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.execution.TrainingExecutionContext
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

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val inputMatrix = FloatMatrix.random(rows, columns, source)
        val variableMatrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext(1, true)
        val inputSource = executionContext.registerMainInputSource(inputMatrix.toTensor(3))
        val optimizer = SimpleGradientDescentOptimizer(inputSource)
        val learningRate = 0.01f

        val expectedResult = inputMatrix + variableMatrix

        val variable = variableMatrix.toVariable(executionContext, optimizer, learningRate)
        val add = Add(inputSource, variable)

        val resultCell = ResultMemoryCellCostFunction(add)
        executionContext.initializeExecution(resultCell)
        executionContext.executePropagation()

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            resultCell.result, 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun leftDifferentiationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val inputMatrix = FloatMatrix.random(rows, columns, source)
        val variableMatrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext(1, rows)
        val inputSource = executionContext.registerMainInputSource(inputMatrix.toTensor(3))
        val optimizer = SimpleGradientDescentOptimizer(inputSource)

        val learningRate = 0.01f

        val variable = variableMatrix.toVariable(executionContext, optimizer, learningRate)

        val add = Add(variable, inputSource)
        val gradientsMatrix = FloatMatrix.random(rows, columns, source)
        val gradients = GradientSource(intArrayOf(rows, columns), gradientsMatrix.toFlatArray(), add)

        executionContext.initializeExecution(gradients)
        executionContext.executePropagation()

        val expectedResult = variableMatrix - gradientsMatrix * learningRate

        Assertions.assertArrayEquals(expectedResult.toFlatArray(), variable.data, 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun rightDifferentiationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val inputMatrix = FloatMatrix.random(rows, columns, source)
        val variableMatrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext(1, rows)
        val inputSource = executionContext.registerMainInputSource(inputMatrix.toTensor(3))

        val optimizer = SimpleGradientDescentOptimizer(inputSource)

        val learningRate = 0.01f
        val variable = variableMatrix.toVariable(executionContext, optimizer, learningRate)

        val add = Add(inputSource, variable)
        val gradients = FloatMatrix.random(rows, columns, source)

        val gradientsSource = GradientSource(intArrayOf(rows, columns), gradients.toFlatArray(), add)
        val expectedResult = variableMatrix - gradients * learningRate

        executionContext.initializeExecution(gradientsSource)
        executionContext.executePropagation()

        Assertions.assertArrayEquals(expectedResult.toFlatArray(), variable.data, 0.001f)
    }
}