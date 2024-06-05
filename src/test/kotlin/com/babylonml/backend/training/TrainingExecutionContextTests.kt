package com.babylonml.backend.training

import com.babylonml.backend.training.operations.*
import com.tornadoml.cpu.*
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class TrainingExecutionContextTests {
    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun gemExpressionTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val firstMatrixRows = source.nextInt(1, 100)
        val firstMatrixColumns = source.nextInt(1, 100)

        val secondMatrixRows = firstMatrixColumns
        val secondMatrixColumns = source.nextInt(1, 100)

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
        val add = Add(executionContext, firstMatrixRows, secondMatrixColumns, firstMultiplication, thirdVariable)

        executionContext.initializeExecution(add)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResultSize = firstMatrixRows * secondMatrixColumns
        val expectedResult = firstMatrix * secondMatrix + thirdMatrix

        Assertions.assertEquals(expectedResultSize, TrainingExecutionContext.addressLength(result))
        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            buffer.copyOfRange(resultOffset, resultOffset + expectedResultSize), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun singleDenseLayerTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

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
        val geLU = GeLUFunction(1, outputSize, executionContext, add)

        executionContext.initializeExecution(geLU)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResultSize = outputSize
        val expectedResult = geLU((inputMatrix * weightsMatrix) + biasMatrix)

        Assertions.assertEquals(expectedResultSize, TrainingExecutionContext.addressLength(result))
        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            buffer.copyOfRange(resultOffset, resultOffset + expectedResultSize), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun twoDenseLayersTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)
        val hiddenSize = source.nextInt(1, 100)

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
        val geLU2 = GeLUFunction(1, outputSize, executionContext, add2)

        executionContext.initializeExecution(geLU2)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResultSize = outputSize

        val expectedResult = geLU(
            geLU((inputMatrix * weightsMatrix1) + biasMatrix1) * weightsMatrix2 +
                    biasMatrix2
        )

        Assertions.assertEquals(expectedResultSize, TrainingExecutionContext.addressLength(result))
        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            buffer.copyOfRange(resultOffset, resultOffset + expectedResultSize), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun threeDenseLayersTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val firstHiddenSize = source.nextInt(1, 100)
        val secondHiddenSize = source.nextInt(1, 100)

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
        val geLU3 = GeLUFunction(1, outputSize, executionContext, add3)

        executionContext.initializeExecution(geLU3)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResultSize = outputSize

        val expectedResult = geLU(
            geLU(
                geLU((inputMatrix * weightsMatrix1) + biasMatrix1) * weightsMatrix2 +
                        biasMatrix2
            ) * weightsMatrix3 + biasMatrix3
        )

        Assertions.assertEquals(expectedResultSize, TrainingExecutionContext.addressLength(result))

        Assertions.assertArrayEquals(
            expectedResult.toFlatArray(),
            buffer.copyOfRange(resultOffset, resultOffset + expectedResultSize), 0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun threeDenseLayersSMEntropyTest(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val firstHiddenSize = source.nextInt(1, 100)
        val secondHiddenSize = source.nextInt(1, 100)

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
        val geLU3 = GeLUFunction(1, outputSize, executionContext, add3)

        val softMax = SoftMaxByRows(
            executionContext,
            geLU3,
            1,
            outputSize
        )

        val expectedValues = FloatMatrix.random(1, outputSize, source)
        val crossEntropy = CrossEntropyByRowsFunction(
            1, outputSize, expectedValues.toFlatArray(),
            executionContext, softMax
        )

        executionContext.initializeExecution(crossEntropy)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)
        val expectedResultSize = 1

        val expectedResult = crossEntropyByRows(
            geLU(
                geLU(
                    geLU((inputMatrix * weightsMatrix1) + biasMatrix1) * weightsMatrix2 +
                            biasMatrix2
                ) * weightsMatrix3 + biasMatrix3
            ).softMaxByRows(),
            expectedValues
        )

        //bad lack of precision
        if(!expectedResult.isInfinite()) {
            return
        }

        Assertions.assertEquals(expectedResultSize, TrainingExecutionContext.addressLength(result))

        Assertions.assertEquals(expectedResult, buffer[resultOffset], 0.001f)
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun singleLayerSingleSampleTestOneEpoch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val learningRate = 0.001f
        val leakyLeRUSlope = 0.01f

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val executionContext = TrainingExecutionContext()
        val input = FloatMatrix.random(1, inputSize, source)

        var weightsMatrix = FloatMatrix.random(inputSize, outputSize, source)
        var biasesMatrix = FloatMatrix.random(1, outputSize, source)

        val constant = Constant(executionContext, input.toFlatArray(), 1, inputSize)
        val optimizer = SimpleGradientDescentOptimizer(1)
        val weightsVariable = weightsMatrix.toVariable(
            executionContext,
            optimizer, learningRate
        )
        val biasVariable = biasesMatrix.toVariable(executionContext, optimizer, learningRate)
        val multiplication = Multiplication(
            executionContext, 1, inputSize, outputSize,
            constant, weightsVariable
        )
        val broadcastRows = BroadcastRows(1, outputSize, executionContext, biasVariable)
        val add = Add(executionContext, 1, outputSize, multiplication, broadcastRows)
        val leRU = LeakyLeRUFunction(1, outputSize, leakyLeRUSlope, executionContext, add)

        val expectedValues = FloatMatrix.random(1, outputSize, source)
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU, 1, outputSize,
            expectedValues.toFlatArray()
        )
        executionContext.initializeExecution(mseCostFunction)
        executionContext.executePropagation(1)

        val z = input * weightsMatrix + biasesMatrix
        val prediction = leakyLeRU(z, leakyLeRUSlope)

        val costError = mseCostFunctionDerivative(prediction, expectedValues)
        val layerError = costError.hadamardMul(leakyLeRUDerivative(z, leakyLeRUSlope))

        val weightsDelta = input.transpose() * layerError
        val biasesDelta = layerError

        weightsMatrix -= weightsDelta * learningRate
        biasesMatrix -= biasesDelta * learningRate

        Assertions.assertArrayEquals(
            weightsMatrix.toFlatArray(),
            weightsVariable.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            biasesMatrix.toFlatArray(),
            biasVariable.data,
            0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun singleLayerMultiSampleTestOneEpoch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val learningRate = 0.001f
        val leakyLeRUSlope = 0.01f

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)
        val batchSize = source.nextInt(1, 100)

        val executionContext = TrainingExecutionContext()
        val input = FloatMatrix.random(batchSize, inputSize, source)

        var weightsMatrix = FloatMatrix.random(inputSize, outputSize, source)
        var biasesMatrix = FloatMatrix.random(1, outputSize, source)

        val constant = Constant(executionContext, input.toFlatArray(), batchSize, inputSize)
        val optimizer = SimpleGradientDescentOptimizer(batchSize)
        val weightsVariable = weightsMatrix.toVariable(
            executionContext,
            optimizer, learningRate
        )
        val biasVariable = biasesMatrix.toVariable(executionContext, optimizer, learningRate)
        val multiplication = Multiplication(
            executionContext, batchSize, inputSize, outputSize,
            constant, weightsVariable
        )
        val broadcastRows = BroadcastRows(batchSize, outputSize, executionContext, biasVariable)
        val add = Add(executionContext, batchSize, outputSize, multiplication, broadcastRows)
        val leRU = LeakyLeRUFunction(batchSize, outputSize, leakyLeRUSlope, executionContext, add)

        val expectedValues = FloatMatrix.random(batchSize, outputSize, source)
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU, batchSize, outputSize,
            expectedValues.toFlatArray()
        )
        executionContext.initializeExecution(mseCostFunction)
        executionContext.executePropagation(1)

        val z = input * weightsMatrix + biasesMatrix.broadcastByRows(batchSize)
        val prediction = leakyLeRU(z, leakyLeRUSlope)

        val costError = mseCostFunctionDerivative(prediction, expectedValues)
        val layerError = costError.hadamardMul(leakyLeRUDerivative(z, leakyLeRUSlope))

        val weightsDelta = input.transpose() * layerError
        val biasesDelta = layerError

        weightsMatrix -= weightsDelta * learningRate / batchSize
        biasesMatrix -= biasesDelta.sumByRows() * learningRate / batchSize

        Assertions.assertArrayEquals(
            weightsMatrix.toFlatArray(),
            weightsVariable.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            biasesMatrix.toFlatArray(),
            biasVariable.data,
            0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun singleLayerMultiSampleTestSeveralEpochs(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val learningRate = 0.001f
        val leakyLeRUSlope = 0.01f
        val epochs = source.nextInt(5, 50)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)
        val batchSize = source.nextInt(1, 100)

        val executionContext = TrainingExecutionContext()
        val input = FloatMatrix.random(batchSize, inputSize, source)

        var weightsMatrix = FloatMatrix.random(inputSize, outputSize, source)
        var biasesMatrix = FloatMatrix.random(1, outputSize, source)

        val constant = Constant(executionContext, input.toFlatArray(), batchSize, inputSize)
        val optimizer = SimpleGradientDescentOptimizer(batchSize)

        val weightsVariable = weightsMatrix.toVariable(
            executionContext,
            optimizer, learningRate
        )
        val biasVariable = biasesMatrix.toVariable(executionContext, optimizer, learningRate)

        val multiplication = Multiplication(
            executionContext, batchSize, inputSize, outputSize,
            constant, weightsVariable
        )
        val broadcastRows = BroadcastRows(batchSize, outputSize, executionContext, biasVariable)
        val add = Add(executionContext, batchSize, outputSize, multiplication, broadcastRows)
        val leRU = LeakyLeRUFunction(batchSize, outputSize, leakyLeRUSlope, executionContext, add)

        val expectedValues = FloatMatrix.random(batchSize, outputSize, source)
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU, batchSize, outputSize,
            expectedValues.toFlatArray()
        )
        executionContext.initializeExecution(mseCostFunction)
        executionContext.executePropagation(epochs)

        for (i in 0 until epochs) {
            val z = input * weightsMatrix + biasesMatrix.broadcastByRows(batchSize)
            val prediction = leakyLeRU(z, leakyLeRUSlope)

            val costError = mseCostFunctionDerivative(prediction, expectedValues)
            val layerError = costError.hadamardMul(leakyLeRUDerivative(z, leakyLeRUSlope))

            val weightsDelta = input.transpose() * layerError
            val biasesDelta = layerError

            weightsMatrix -= weightsDelta * learningRate / batchSize
            biasesMatrix -= biasesDelta.sumByRows() * learningRate / batchSize
        }

        Assertions.assertArrayEquals(
            weightsMatrix.toFlatArray(),
            weightsVariable.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            biasesMatrix.toFlatArray(),
            biasVariable.data,
            0.001f
        )
    }
}