package com.babylonml.backend.training.operations

import com.babylonml.backend.training.optimizer.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import com.babylonml.matrix.FloatMatrix
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.leakyLeRU
import com.babylonml.leakyLeRUDerivative
import it.unimi.dsi.fastutil.ints.IntImmutableList
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class LeakyLeRUTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val matrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext(1, rows)
        val inputSource = executionContext.registerMainInputSource(matrix.toTensor(3))
        val leakyLeRUSlope = 0.01f

        val leakyLeRU = LeakyLeRUFunction(leakyLeRUSlope, inputSource)

        val resultCell = ResultMemoryCellCostFunction(leakyLeRU)
        executionContext.initializeExecution(resultCell)

        executionContext.executePropagation()

        val expectedResult = leakyLeRU(matrix, leakyLeRUSlope)

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            resultCell.result, 0.001f
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
        val executionContext = TrainingExecutionContext(1, rows)
        val inputSource = executionContext.registerMainInputSource(inputMatrix.toTensor(3))

        val optimizer = SimpleGradientDescentOptimizer(inputSource)

        val learningRate = 0.01f
        val leakyLeRUSlope = 0.01f

        val variable = matrix.toVariable(executionContext, optimizer, learningRate)
        val add = Add(inputSource, variable)
        val leakyLeRU = LeakyLeRUFunction(leakyLeRUSlope, add)

        val gradients = FloatMatrix.random(rows, columns, source)
        val gradientSource = GradientSource(IntImmutableList.of(rows, columns), gradients.toFlatArray(), leakyLeRU)

        executionContext.initializeExecution(gradientSource)
        executionContext.executePropagation()

        val resultGradient = leakyLeRUDerivative(matrix, leakyLeRUSlope).hadamardMul(gradients)
        val expectedResult = matrix - resultGradient * learningRate

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            variable.data,
            0.001f
        )
    }
}