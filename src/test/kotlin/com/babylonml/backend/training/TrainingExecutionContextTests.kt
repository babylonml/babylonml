package com.babylonml.backend.training

import com.babylonml.backend.training.operations.Add
import com.babylonml.backend.training.operations.GeLUFunction
import com.babylonml.backend.training.operations.Multiplication
import com.tornadoml.cpu.FloatMatrix
import com.tornadoml.cpu.SeedsArgumentsProvider
import com.tornadoml.cpu.geLU
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class TrainingExecutionContextTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun gemExpressionTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val firstMatrixRows = source.nextInt(100)
        val firstMatrixColumns = source.nextInt(100)

        val secondMatrixRows = firstMatrixColumns
        val secondMatrixColumns = source.nextInt(100)

        val thirdMatrixRows = firstMatrixRows
        val thirdMatrixColumns = secondMatrixColumns

        val firstMatrix = FloatMatrix.random(firstMatrixRows, firstMatrixColumns, source)
        val secondMatrix = FloatMatrix.random(secondMatrixRows, secondMatrixColumns, source)
        val thirdMatrix = FloatMatrix.random(thirdMatrixRows, thirdMatrixColumns, source)

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val firstVariable = firstMatrix.toVariable(executionContext, optimizer, learningRate)
        val secondVariable = secondMatrix.toVariable(executionContext, optimizer, learningRate)
        val thirdVariable = thirdMatrix.toVariable(executionContext, optimizer, learningRate)

        val firstMultiplication = Multiplication(
            executionContext, firstMatrixRows, firstMatrixColumns, secondMatrixColumns,
            firstVariable, secondVariable
        )
        Add(executionContext, firstMatrixRows, secondMatrixColumns, firstMultiplication, thirdVariable)

        executionContext.initializeExecution()

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result[0])
        val resultOffset = TrainingExecutionContext.addressOffset(result[0])

        val expectedResultSize = firstMatrixRows * secondMatrixColumns
        val expectedResult = firstMatrix * secondMatrix + thirdMatrix

        Assertions.assertEquals(1, result.size)
        Assertions.assertEquals(expectedResultSize, TrainingExecutionContext.addressLength(result[0]))
        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            buffer.copyOfRange(resultOffset, resultOffset + expectedResultSize), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun singleDenseLayerTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(100)
        val outputSize = source.nextInt(100)

        val inputMatrix = FloatMatrix.random(1, inputSize, source)
        val weightsMatrix = FloatMatrix.random(inputSize, outputSize, source)
        val biasMatrix = FloatMatrix.random(1, outputSize, source)

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val inputVariable = inputMatrix.toVariable(executionContext, optimizer, learningRate)
        val weightsVariable = weightsMatrix.toVariable(executionContext, optimizer, learningRate)
        val biasVariable = biasMatrix.toVariable(executionContext, optimizer, learningRate)

        val multiplication = Multiplication(
            executionContext, 1, inputSize, outputSize,
            inputVariable, weightsVariable
        )
        val add = Add(executionContext, 1, outputSize, multiplication, biasVariable)
        GeLUFunction(1, outputSize, executionContext, add)

        executionContext.initializeExecution()

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result[0])
        val resultOffset = TrainingExecutionContext.addressOffset(result[0])

        val expectedResultSize = outputSize
        val expectedResult = geLU((inputMatrix * weightsMatrix) + biasMatrix)

        Assertions.assertEquals(1, result.size)
        Assertions.assertEquals(expectedResultSize, TrainingExecutionContext.addressLength(result[0]))
        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            buffer.copyOfRange(resultOffset, resultOffset + expectedResultSize), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun twoDenseLayersTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(100)
        val outputSize = source.nextInt(100)
        val hiddenSize = source.nextInt(100)

        val inputMatrix = FloatMatrix.random(1, inputSize, source)
        val weightsMatrix1 = FloatMatrix.random(inputSize, hiddenSize, source)
        val biasMatrix1 = FloatMatrix.random(1, hiddenSize, source)

        val weightsMatrix2 = FloatMatrix.random(hiddenSize, outputSize, source)
        val biasMatrix2 = FloatMatrix.random(1, outputSize, source)

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val inputVariable = inputMatrix.toVariable(executionContext, optimizer, learningRate)
        val weightsVariable1 = weightsMatrix1.toVariable(executionContext, optimizer, learningRate)
        val biasVariable1 = biasMatrix1.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable2 = weightsMatrix2.toVariable(executionContext, optimizer, learningRate)
        val biasVariable2 = biasMatrix2.toVariable(executionContext, optimizer, learningRate)

        val multiplication1 = Multiplication(
            executionContext, 1, inputSize, hiddenSize,
            inputVariable, weightsVariable1
        )
        val add1 = Add(executionContext, 1, hiddenSize, multiplication1, biasVariable1)
        val geLU1 = GeLUFunction(1, hiddenSize, executionContext, add1)

        val multiplication2 = Multiplication(
            executionContext, 1, hiddenSize, outputSize,
            geLU1, weightsVariable2
        )
        val add2 = Add(executionContext, 1, outputSize, multiplication2, biasVariable2)
        GeLUFunction(1, outputSize, executionContext, add2)

        executionContext.initializeExecution()

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result[0])
        val resultOffset = TrainingExecutionContext.addressOffset(result[0])

        val expectedResultSize = outputSize

        val expectedResult = geLU(geLU((inputMatrix * weightsMatrix1) + biasMatrix1) * weightsMatrix2 +
                biasMatrix2)

        Assertions.assertEquals(1, result.size)
        Assertions.assertEquals(expectedResultSize, TrainingExecutionContext.addressLength(result[0]))
        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            buffer.copyOfRange(resultOffset, resultOffset + expectedResultSize), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun threeDenseLayersTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(100)
        val outputSize = source.nextInt(100)

        val firstHiddenSize = source.nextInt(100)
        val secondHiddenSize = source.nextInt(100)

        val inputMatrix = FloatMatrix.random(1, inputSize, source)
        val weightsMatrix1 = FloatMatrix.random(inputSize, firstHiddenSize, source)
        val biasMatrix1 = FloatMatrix.random(1, firstHiddenSize, source)

        val weightsMatrix2 = FloatMatrix.random(firstHiddenSize, secondHiddenSize, source)
        val biasMatrix2 = FloatMatrix.random(1, secondHiddenSize, source)

        val weightsMatrix3 = FloatMatrix.random(secondHiddenSize, outputSize, source)
        val biasMatrix3 = FloatMatrix.random(1, outputSize, source)

        val executionContext = TrainingExecutionContext()
        val optimizer = SimpleGradientDescentOptimizer(1)
        val learningRate = 0.01f

        val inputVariable = inputMatrix.toVariable(executionContext, optimizer, learningRate)
        val weightsVariable1 = weightsMatrix1.toVariable(executionContext, optimizer, learningRate)
        val biasVariable1 = biasMatrix1.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable2 = weightsMatrix2.toVariable(executionContext, optimizer, learningRate)
        val biasVariable2 = biasMatrix2.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable3 = weightsMatrix3.toVariable(executionContext, optimizer, learningRate)
        val biasVariable3 = biasMatrix3.toVariable(executionContext, optimizer, learningRate)

        val multiplication1 = Multiplication(
            executionContext, 1, inputSize, firstHiddenSize,
            inputVariable, weightsVariable1
        )
        val add1 = Add(executionContext, 1, firstHiddenSize, multiplication1, biasVariable1)
        val geLU1 = GeLUFunction(1, firstHiddenSize, executionContext, add1)

        val multiplication2 = Multiplication(
            executionContext, 1, firstHiddenSize, secondHiddenSize,
            geLU1, weightsVariable2
        )
        val add2 = Add(executionContext, 1, secondHiddenSize, multiplication2, biasVariable2)
        val geLU2 = GeLUFunction(1, secondHiddenSize, executionContext, add2)

        val multiplication3 = Multiplication(
            executionContext, 1, secondHiddenSize, outputSize,
            geLU2, weightsVariable3
        )

        val add3 = Add(executionContext, 1, outputSize, multiplication3, biasVariable3)
        GeLUFunction(1, outputSize, executionContext, add3)

        executionContext.initializeExecution()

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result[0])
        val resultOffset = TrainingExecutionContext.addressOffset(result[0])

        val expectedResultSize = outputSize

        val expectedResult = geLU(geLU(geLU((inputMatrix * weightsMatrix1) + biasMatrix1) * weightsMatrix2 +
                biasMatrix2) * weightsMatrix3 + biasMatrix3)

        Assertions.assertEquals(1, result.size)
        Assertions.assertEquals(expectedResultSize, TrainingExecutionContext.addressLength(result[0]))

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            buffer.copyOfRange(resultOffset, resultOffset + expectedResultSize), 0.005f
        )
    }
}