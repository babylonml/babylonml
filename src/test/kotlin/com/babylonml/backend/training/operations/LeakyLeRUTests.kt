package com.babylonml.backend.training.operations

import com.babylonml.backend.training.optimizer.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.babylonml.matrix.FloatMatrix
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.leakyLeRU
import com.babylonml.leakyLeRUDerivative
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class LeakyLeRUTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(100)
        val columns = source.nextInt(100)

        val matrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext()
        val optimizer =
            SimpleGradientDescentOptimizer(NullDataSource())
        val learningRate = 0.01f

        val leakyLeRUSlope = 0.01f

        val variable = matrix.toVariable(executionContext, optimizer, learningRate)
        val leakyLeRU = LeakyLeRUFunction(leakyLeRUSlope, executionContext, variable)

        executionContext.initializeExecution(leakyLeRU)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResult = leakyLeRU(matrix, leakyLeRUSlope)

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

        val matrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext()
        val optimizer =
            SimpleGradientDescentOptimizer(NullDataSource())

        val learningRate = 0.01f
        val leakyLeRUSlope = 0.01f

        val variable = matrix.toVariable(executionContext, optimizer, learningRate)
        val leakyLeRU = LeakyLeRUFunction(leakyLeRUSlope, executionContext, variable)

        val gradients = FloatMatrix.random(rows, columns, source)
        val gradientSource = GradientSource(executionContext, rows, columns, gradients.toFlatArray(), leakyLeRU)

        executionContext.initializeExecution(gradientSource)
        executionContext.executePropagation(1)

        val resultGradient = leakyLeRUDerivative(matrix, leakyLeRUSlope).hadamardMul(gradients)
        val expectedResult = matrix - resultGradient * learningRate

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            variable.data,
            0.001f
        )
    }
}