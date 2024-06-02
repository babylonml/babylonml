package com.babylonml.backend.training.operations

import com.babylonml.backend.training.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.tornadoml.cpu.FloatMatrix
import com.tornadoml.cpu.SeedsArgumentsProvider
import com.tornadoml.cpu.geLU
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class GeLUTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(100)
        val  columns = source.nextInt(100)

        val matrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val variable = matrix.toVariable(executionContext, optimizer, learningRate)
        GeLUFunction(rows, columns, executionContext, variable)

        executionContext.initializeExecution()

        val result = executionContext.executeForwardPropagation()
        Assertions.assertEquals(1, result.size)

        val buffer = executionContext.getMemoryBuffer(result[0])
        val resultOffset = TrainingExecutionContext.addressOffset(result[0])

        val expectedResult = geLU(matrix)

        Assertions.assertArrayEquals(expectedResult.toFlatArray(),
            buffer.copyOfRange(resultOffset, resultOffset + rows * columns), 0.001f)
    }
}