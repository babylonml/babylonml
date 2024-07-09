package com.babylonml.backend.training.operations

import com.babylonml.backend.training.optimizer.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import com.babylonml.matrix.FloatMatrix
import com.babylonml.SeedsArgumentsProvider
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class HadamardProductTests {
    fun forwardTest(seed: Long) {
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
        val hadamard = HadamardProduct(inputSource, variable)

        val resultCell = ResultMemoryCellCostFunction(hadamard)
        executionContext.initializeExecution(resultCell)
        executionContext.executePropagation()

        val expectedResult = inputMatrix.hadamardMul(variableMatrix)

        Assertions.assertArrayEquals(expectedResult.toFlatArray(),resultCell.result, 0.001f)
    }

    fun differentiationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val inputMatrix = FloatMatrix(rows, columns)
        val firstMatrix = FloatMatrix.random(rows, columns, source)
        val secondMatrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext(1)
        val inputSource = executionContext.registerMainInputSource(inputMatrix.toTensor(3))
        val optimizer = SimpleGradientDescentOptimizer(inputSource)

        val learningRate = 0.01f

        val firstVariable = firstMatrix.toVariable("first", executionContext, optimizer, learningRate)
        val secondVariable = secondMatrix.toVariable("second", executionContext, optimizer, learningRate)

        val add = Add(inputSource, firstVariable)
        val hadamard = HadamardProduct(add, secondVariable)
        val gradients = FloatMatrix.random(rows, columns, source)

        val gradientSource = GradientSource(gradients.toTensor(), hadamard)

        val firstGradient = gradients.hadamardMul(secondMatrix)
        val secondGradient = gradients.hadamardMul(firstMatrix)

        val expectedFirstResult = firstMatrix - firstGradient * learningRate
        val expectedSecondResult = secondMatrix - secondGradient * learningRate

        executionContext.initializeExecution(gradientSource)
        executionContext.executePropagation()

        Assertions.assertArrayEquals(
            expectedFirstResult.toFlatArray(),
            firstVariable.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            expectedSecondResult.toFlatArray(),
            secondVariable.data,
            0.001f
        )
    }
}