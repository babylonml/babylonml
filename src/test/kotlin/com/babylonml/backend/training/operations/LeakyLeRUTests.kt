package com.babylonml.backend.training.operations

import com.babylonml.backend.training.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.tornadoml.cpu.FloatMatrix
import com.tornadoml.cpu.SeedsArgumentsProvider
import com.tornadoml.cpu.leakyLeRU
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
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val leakyLeRUSlope = 0.01f

        val variable = matrix.toVariable(executionContext, optimizer, learningRate)
        val leakyLeRU = LeakyLeRUFunction(rows, columns, leakyLeRUSlope, executionContext, variable)

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
}