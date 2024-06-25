package com.babylonml.backend.training.operations

import com.babylonml.backend.training.optimizer.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import com.babylonml.matrix.FloatMatrix
import com.babylonml.SeedsArgumentsProvider
import com.babylonml.mseCostFunction
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class MSECostFunctionTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun forwardTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val predictedValuesMatrix = FloatMatrix.random(rows, columns, source)
        val expectedValuesMatrix = FloatMatrix.random(
            rows, columns, source
        )

        val executionContext = TrainingExecutionContext(1, rows)
        val inputSource = executionContext.registerMainInputSource(predictedValuesMatrix.toTensor())

        val expectedValuesConst = Constant(
            executionContext, expectedValuesMatrix.toTensor()
        )
        val mseCostFunction = MSECostFunction(inputSource, expectedValuesConst)

        executionContext.initializeExecution(mseCostFunction)

        var result = 0.0f
        executionContext.executePropagation { _, cost ->
            result = cost
        }

        val expectedResult = mseCostFunction(predictedValuesMatrix, expectedValuesMatrix) / rows
        Assertions.assertEquals(expectedResult, result, 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun differentiationTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val inputMatrix = FloatMatrix(rows, columns)
        val predictedValuesMatrix = FloatMatrix.random(rows, columns, source)
        val expectedValuesMatrix = FloatMatrix.random(
            rows, columns, source
        )

        val executionContext = TrainingExecutionContext(1, rows)
        val inputSource = executionContext.registerMainInputSource(inputMatrix.toTensor(3))
        val optimizer = SimpleGradientDescentOptimizer(inputSource)
        val learningRate = 0.01f

        val predictedValuesVariable = predictedValuesMatrix.toVariable(
            executionContext, optimizer, learningRate
        )

        val add = Add(inputSource, predictedValuesVariable)
        val expectedValues = executionContext.registerAdditionalInputSource(expectedValuesMatrix.toTensor(3))
        val mseCostFunction = MSECostFunction(add, expectedValues)

        executionContext.initializeExecution(mseCostFunction)
        executionContext.executePropagation()

        val expectedGradients = predictedValuesMatrix - expectedValuesMatrix

        val expectedResult = predictedValuesMatrix - expectedGradients * learningRate
        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            predictedValuesVariable.data,
            0.001f
        )
    }
}