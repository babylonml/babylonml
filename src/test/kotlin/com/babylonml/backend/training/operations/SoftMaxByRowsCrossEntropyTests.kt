package com.babylonml.backend.training.operations

import com.babylonml.backend.training.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.tornadoml.cpu.FloatMatrix
import com.tornadoml.cpu.SeedsArgumentsProvider
import com.tornadoml.cpu.crossEntropyByRows
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class SoftMaxByRowsCrossEntropyTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val samplesCount = source.nextInt(1, 100)

        val samplesMatrix = FloatMatrix.random(inputSize, samplesCount, source)
        val expectedProbabilitiesMatrix = FloatMatrix.random(
            inputSize, samplesCount,
            0f, 1f, source
        )

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val variable = samplesMatrix.toVariable(executionContext, optimizer, learningRate)
        val softMax = SoftMaxByRows(
            executionContext,
            variable,
            inputSize,
            samplesCount
        )
        val crossEntropy = CrossEntropyByRowsFunction(
            inputSize, samplesCount, expectedProbabilitiesMatrix.toFlatArray(),
            executionContext, softMax
        )

        executionContext.initializeExecution(crossEntropy)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResult = crossEntropyByRows(
            samplesMatrix.softMaxByRows(),
            expectedProbabilitiesMatrix
        )

        Assertions.assertEquals(expectedResult, buffer[resultOffset], 0.001f)
    }
}