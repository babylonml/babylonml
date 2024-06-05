package com.babylonml.backend.training.operations

import com.babylonml.backend.training.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.tornadoml.cpu.FloatMatrix
import com.tornadoml.cpu.SeedsArgumentsProvider
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class HadamardProductTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val firstMatrix = FloatMatrix.random(rows, columns, source)
        val secondMatrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val firstVariable = firstMatrix.toVariable(executionContext, optimizer, learningRate)
        val secondVariable = secondMatrix.toVariable(executionContext, optimizer, learningRate)

        val hadamard = HadamardProduct(rows, columns, executionContext, firstVariable, secondVariable)

        executionContext.initializeExecution(hadamard)
        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResult = firstMatrix.hadamardMul(secondMatrix)

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            buffer.copyOfRange(resultOffset, resultOffset + rows * columns), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun differentiationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val firstMatrix = FloatMatrix.random(rows, columns, source)
        val secondMatrix = FloatMatrix.random(rows, columns, source)

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val firstVariable = firstMatrix.toVariable(executionContext, optimizer, learningRate)
        val secondVariable = secondMatrix.toVariable(executionContext, optimizer, learningRate)

        val hadamard = HadamardProduct(rows, columns, executionContext, firstVariable, secondVariable)
        val gradients = FloatMatrix.random(rows, columns, source)

        val gradientSource = GradientSource(executionContext, rows, columns, gradients.toFlatArray(), hadamard)

        executionContext.initializeExecution(gradientSource)
        executionContext.executePropagation(1)

        val firstGradient = gradients.hadamardMul(secondMatrix)
        val secondGradient = gradients.hadamardMul(firstMatrix)

        val expectedFirstResult = firstMatrix - firstGradient * learningRate
        val expectedSecondResult = secondMatrix - secondGradient * learningRate

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