package com.babylonml.backend.training.operations

import com.babylonml.backend.training.optimizer.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import com.babylonml.matrix.FloatMatrix
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.geLU
import com.babylonml.geLUDerivative
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class GeLUTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val matrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext(1, rows)
        val inputSource = executionContext.registerMainInputSource(matrix.toTensor())

        val geLU = GeLUFunction(inputSource)

        val memoryCell = ResultMemoryCellCostFunction(geLU)
        executionContext.initializeExecution(memoryCell)

        executionContext.executePropagation()

        val expectedResult = geLU(matrix)

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            memoryCell.result, 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun differentiationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val matrix = FloatMatrix.random(rows, columns, source)
        val inputMatrix = FloatMatrix(rows, columns)

        val executionContext = TrainingExecutionContext(1)
        val inputSource = executionContext.registerMainInputSource(inputMatrix.toTensor())

        val learningRate = 0.01f
        val optimizer = SimpleGradientDescentOptimizer(inputSource)
        val variable = matrix.toVariable(executionContext, optimizer, learningRate)

        val add = Add(inputSource, variable)
        val geLU = GeLUFunction(add)

        val gradients = FloatMatrix.random(rows, columns, source)
        val gradientSource = GradientSource(intArrayOf(rows, columns), gradients.toFlatArray(), geLU)

        executionContext.initializeExecution(gradientSource)
        executionContext.executePropagation()

        val expectedGradients = geLUDerivative(matrix).hadamardMul(gradients)

        val expectedResult = matrix - expectedGradients * learningRate / rows

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(), variable.data, 0.001f
        )
    }
}