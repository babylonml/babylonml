package com.babylonml.backend.training.operations

import com.babylonml.backend.training.optimizer.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.execution.TrainingExecutionContext
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

        val executionContext = TrainingExecutionContext(1, rows)
        val inputSource = executionContext.registerMainInputSource(matrix.toArray())
        val expectedProbabilitiesSource =
            executionContext.registerAdditionalInputSource(expectedProbabilitiesMatrix.toArray())

        val softMax = SoftMaxByRows(inputSource)
        val crossEntropy = CrossEntropyCostFunction(
            expectedProbabilitiesSource, softMax
        )

        executionContext.initializeExecution(crossEntropy)
        var result = 0.0f

        executionContext.executePropagation { _, cost ->
            result = cost
        }

        val expectedResult = crossEntropyByRows(
            matrix.softMaxByRows(),
            expectedProbabilitiesMatrix
        )

        Assertions.assertEquals(expectedResult, result, 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun differentiationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)
        val matrix = FloatMatrix.random(rows, columns, source)
        val inputMatrix = FloatMatrix(rows, columns)

        val expectedMatrix = FloatMatrix.random(
            rows, columns,
            0f, 1f, source
        )

        val executionContext = TrainingExecutionContext(1)
        val inputSource = executionContext.registerMainInputSource(inputMatrix.toArray())
        val expectedSource = executionContext.registerAdditionalInputSource(expectedMatrix.toArray())
        val optimizer = SimpleGradientDescentOptimizer(inputSource)

        val learningRate = 0.01f

        val variable = matrix.toVariable(executionContext, optimizer, learningRate)
        val add = Add(inputSource, variable)
        val softMax = SoftMaxByRows(add)

        val crossEntropy = CrossEntropyCostFunction(expectedSource, softMax)

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