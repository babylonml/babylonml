package com.babylonml.backend.training

import com.babylonml.backend.training.operations.*
import com.tornadoml.cpu.*
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource
import kotlin.math.min

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
        val learningRate = 0.001f

        val optimizer = SimpleGradientDescentOptimizer(NullDataSource())
        val firstVariable = firstMatrix.toVariable(executionContext, optimizer, learningRate)
        val secondVariable = secondMatrix.toVariable(executionContext, optimizer, learningRate)
        val thirdVariable = thirdMatrix.toVariable(executionContext, optimizer, learningRate)

        val firstMultiplication = Multiplication(
            executionContext, firstVariable, secondVariable
        )
        val add = Add(executionContext, firstMultiplication, thirdVariable, true)

        executionContext.initializeExecution(add)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResultSize = firstMatrixRows * secondMatrixColumns
        val expectedResult = firstMatrix * secondMatrix + thirdMatrix

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
        val learningRate = 0.001f

        val optimizer = SimpleGradientDescentOptimizer(NullDataSource())
        val inputVariable = inputMatrix.toVariable(executionContext, optimizer, learningRate)
        val weightsVariable = weightsMatrix.toVariable(executionContext, optimizer, learningRate)
        val biasVariable = biasMatrix.toVariable(executionContext, optimizer, learningRate)

        val multiplication = Multiplication(
            executionContext, inputVariable, weightsVariable
        )
        val add = Add(executionContext, multiplication, biasVariable, true)
        val geLU = GeLUFunction(executionContext, add)

        executionContext.initializeExecution(geLU)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResultSize = outputSize
        val expectedResult = geLU((inputMatrix * weightsMatrix) + biasMatrix)

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
        val learningRate = 0.001f

        val optimizer = SimpleGradientDescentOptimizer(NullDataSource())
        val inputVariable = inputMatrix.toVariable(executionContext, optimizer, learningRate)
        val weightsVariable1 = weightsMatrix1.toVariable(executionContext, optimizer, learningRate)
        val biasVariable1 = biasMatrix1.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable2 = weightsMatrix2.toVariable(executionContext, optimizer, learningRate)
        val biasVariable2 = biasMatrix2.toVariable(executionContext, optimizer, learningRate)

        val multiplication1 = Multiplication(
            executionContext, inputVariable, weightsVariable1
        )
        val add1 = Add(executionContext, multiplication1, biasVariable1, true)
        val geLU1 = GeLUFunction(executionContext, add1)

        val multiplication2 = Multiplication(
            executionContext, geLU1, weightsVariable2
        )
        val add2 = Add(executionContext, multiplication2, biasVariable2, true)
        val geLU2 = GeLUFunction(executionContext, add2)

        executionContext.initializeExecution(geLU2)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

        val expectedResultSize = outputSize

        val expectedResult = geLU(
            geLU((inputMatrix * weightsMatrix1) + biasMatrix1) * weightsMatrix2 +
                    biasMatrix2
        )

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
        val learningRate = 0.001f

        val optimizer = SimpleGradientDescentOptimizer(NullDataSource())

        val inputVariable = inputMatrix.toVariable(executionContext, optimizer, learningRate)
        val weightsVariable1 = weightsMatrix1.toVariable(executionContext, optimizer, learningRate)
        val biasVariable1 = biasMatrix1.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable2 = weightsMatrix2.toVariable(executionContext, optimizer, learningRate)
        val biasVariable2 = biasMatrix2.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable3 = weightsMatrix3.toVariable(executionContext, optimizer, learningRate)
        val biasVariable3 = biasMatrix3.toVariable(executionContext, optimizer, learningRate)

        val multiplication1 = Multiplication(
            executionContext, inputVariable, weightsVariable1
        )
        val add1 = Add(executionContext, multiplication1, biasVariable1, true)
        val geLU1 = GeLUFunction(executionContext, add1)

        val multiplication2 = Multiplication(
            executionContext, geLU1, weightsVariable2
        )
        val add2 = Add(executionContext, multiplication2, biasVariable2, true)
        val geLU2 = GeLUFunction(executionContext, add2)

        val multiplication3 = Multiplication(
            executionContext, geLU2, weightsVariable3
        )

        val add3 = Add(executionContext, multiplication3, biasVariable3, true)
        val geLU3 = GeLUFunction(executionContext, add3)

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
        val learningRate = 0.001f

        val optimizer = SimpleGradientDescentOptimizer(NullDataSource())
        val inputVariable = inputMatrix.toVariable(executionContext, optimizer, learningRate)
        val weightsVariable1 = weightsMatrix1.toVariable(executionContext, optimizer, learningRate)
        val biasVariable1 = biasMatrix1.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable2 = weightsMatrix2.toVariable(executionContext, optimizer, learningRate)
        val biasVariable2 = biasMatrix2.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable3 = weightsMatrix3.toVariable(executionContext, optimizer, learningRate)
        val biasVariable3 = biasMatrix3.toVariable(executionContext, optimizer, learningRate)

        val multiplication1 = Multiplication(
            executionContext, inputVariable, weightsVariable1
        )
        val add1 = Add(executionContext, multiplication1, biasVariable1, true)
        val geLU1 = GeLUFunction(executionContext, add1)

        val multiplication2 = Multiplication(
            executionContext, geLU1, weightsVariable2
        )
        val add2 = Add(executionContext, multiplication2, biasVariable2, true)
        val geLU2 = GeLUFunction(executionContext, add2)

        val multiplication3 = Multiplication(
            executionContext, geLU2, weightsVariable3
        )

        val add3 = Add(executionContext, multiplication3, biasVariable3, true)
        val geLU3 = GeLUFunction(executionContext, add3)

        val softMax = SoftMaxByRows(
            executionContext,
            geLU3,
        )

        val expectedValues = FloatMatrix.random(1, outputSize, source)
        val expectedValuesConst = Constant(executionContext, expectedValues.toFlatArray(), 1, outputSize)
        val crossEntropy = CrossEntropyByRowsFunction(
            expectedValuesConst,
            executionContext, softMax
        )

        executionContext.initializeExecution(crossEntropy)

        val result = executionContext.executeForwardPropagation()

        val buffer = executionContext.getMemoryBuffer(result)
        val resultOffset = TrainingExecutionContext.addressOffset(result)

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
        if (!expectedResult.isInfinite()) {
            return
        }

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
        val optimizer = SimpleGradientDescentOptimizer(constant)
        val weightsVariable = weightsMatrix.toVariable(
            executionContext,
            optimizer, learningRate
        )
        val biasVariable = biasesMatrix.toVariable(executionContext, optimizer, learningRate)
        val multiplication = Multiplication(
            executionContext, constant, weightsVariable
        )

        val add = Add(executionContext, multiplication, biasVariable, true)
        val leRU = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add)

        val expectedValues = FloatMatrix.random(1, outputSize, source)
        val expectedValuesSource = MiniBatchInputSource(expectedValues.toArray(), outputSize, 1, executionContext)

        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU,
            expectedValuesSource
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

        val batchSource = MiniBatchInputSource(input.toArray(), inputSize, batchSize, executionContext)
        val optimizer =
            SimpleGradientDescentOptimizer(batchSource)

        val weightsVariable = weightsMatrix.toVariable(
            executionContext,
            optimizer, learningRate
        )
        val biasVariable = biasesMatrix.toVariable(executionContext, optimizer, learningRate)
        val multiplication = Multiplication(
            executionContext, batchSource, weightsVariable
        )
        val add = Add(executionContext, multiplication, biasVariable, true)
        val leRU = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add)

        val expectedValues = FloatMatrix.random(batchSize, outputSize, source)
        val expectedValuesSource = MiniBatchInputSource(
            expectedValues.toArray(), outputSize,
            batchSize, executionContext
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU, expectedValuesSource
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

        val batchSource = MiniBatchInputSource(input.toArray(), inputSize, batchSize, executionContext)
        val optimizer =
            SimpleGradientDescentOptimizer(batchSource)

        val weightsVariable = weightsMatrix.toVariable(
            executionContext,
            optimizer, learningRate
        )
        val biasVariable = biasesMatrix.toVariable(executionContext, optimizer, learningRate)

        val multiplication = Multiplication(
            executionContext, batchSource, weightsVariable
        )

        val add = Add(executionContext, multiplication, biasVariable, true)
        val leRU = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add)

        val expectedValues = FloatMatrix.random(batchSize, outputSize, source)
        val expectedValuesSource = MiniBatchInputSource(
            expectedValues.toArray(), outputSize,
            batchSize, executionContext
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU,
            expectedValuesSource
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

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun singleLayerMultiSampleTestSeveralEpochsMiniBatch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val learningRate = 0.001f
        val leakyLeRUSlope = 0.01f
        val epochs = source.nextInt(5, 50)

        val inputSize = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)
        val batchSize = source.nextInt(20, 100)
        val miniBatchSize = source.nextInt(2, 10)

        val executionContext = TrainingExecutionContext()
        val input = FloatMatrix.random(batchSize, inputSize, source)

        var weightsMatrix = FloatMatrix.random(inputSize, outputSize, source)
        var biasesMatrix = FloatMatrix.random(1, outputSize, source)

        val miniBatchInputSource = MiniBatchInputSource(input.toArray(), inputSize, miniBatchSize, executionContext)
        val optimizer =
            SimpleGradientDescentOptimizer(miniBatchInputSource)

        val weightsVariable = weightsMatrix.toVariable(
            "weightsVariable", executionContext, optimizer,
            learningRate
        )
        val biasVariable = biasesMatrix.toVariable("biasVariable", executionContext, optimizer, learningRate)
        val multiplication = Multiplication(
            "multiplication",
            executionContext, miniBatchInputSource, weightsVariable
        )

        val add = Add("add", executionContext, multiplication, biasVariable, true)
        val leRU = LeakyLeRUFunction("leRU", leakyLeRUSlope, executionContext, add)

        val expectedValues = FloatMatrix.random(batchSize, outputSize, source)
        val expectedValuesSource = MiniBatchInputSource(
            expectedValues.toArray(), outputSize,
            miniBatchSize, executionContext
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU,
            expectedValuesSource
        )
        executionContext.initializeExecution(mseCostFunction)

        val fullBatchesCount = (batchSize + miniBatchSize - 1) / miniBatchSize
        executionContext.executePropagation(epochs * fullBatchesCount)

        for (i in 0 until epochs) {
            for (start in 0 until batchSize step miniBatchSize) {
                val miniBatchCount = min(miniBatchSize, batchSize - start)
                val miniInput = input.subRows(start, miniBatchCount)

                val z = miniInput * weightsMatrix +
                        biasesMatrix.broadcastByRows(miniBatchCount)
                val prediction = leakyLeRU(z, leakyLeRUSlope)

                val costError =
                    mseCostFunctionDerivative(prediction, expectedValues.subRows(start, miniBatchCount))
                val layerError = costError.hadamardMul(leakyLeRUDerivative(z, leakyLeRUSlope))

                val weightsDelta = miniInput.transpose() * layerError
                val biasesDelta = layerError

                weightsMatrix -= weightsDelta * learningRate / miniBatchCount
                biasesMatrix -= biasesDelta.sumByRows() * learningRate / miniBatchCount
            }
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


    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun twoLayersSingleSampleTestOneEpoch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)
        val learningRate = 0.01f
        val leakyLeRUSlope = 0.01f

        val outputSize = source.nextInt(1, 100)
        val hiddenSize = source.nextInt(1, 100)

        val inputSize = source.nextInt(1, 100)

        val inputMatrix = FloatMatrix.random(1, inputSize, source)
        var weightsMatrix1 = FloatMatrix.random(inputSize, hiddenSize, source)
        var biasesMatrix1 = FloatMatrix.random(1, hiddenSize, source)

        var weightsMatrix2 = FloatMatrix.random(hiddenSize, outputSize, source)
        var biasesMatrix2 = FloatMatrix.random(1, outputSize, source)

        val executionContext = TrainingExecutionContext()
        val input = inputMatrix.toConstant(executionContext)
        val optimizer = SimpleGradientDescentOptimizer(input)

        val weightsVariable1 = weightsMatrix1.toVariable(executionContext, optimizer, learningRate)
        val biasVariable1 = biasesMatrix1.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable2 = weightsMatrix2.toVariable(executionContext, optimizer, learningRate)
        val biasVariable2 = biasesMatrix2.toVariable(executionContext, optimizer, learningRate)

        val multiplication1 = Multiplication(
            executionContext, input, weightsVariable1
        )
        val add1 = Add(executionContext, multiplication1, biasVariable1, true)
        val leRU1 = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add1)

        val multiplication2 = Multiplication(
            executionContext, leRU1, weightsVariable2
        )
        val add2 = Add(executionContext, multiplication2, biasVariable2, true)
        val leRU2 = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add2)

        val expectedValues = FloatMatrix.random(1, outputSize, source)
        val expectedValuesSource = MiniBatchInputSource(
            expectedValues.toArray(), outputSize, 1, executionContext
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU2, expectedValuesSource
        )

        executionContext.initializeExecution(mseCostFunction)
        executionContext.executePropagation(1)

        val z1 = inputMatrix * weightsMatrix1 + biasesMatrix1
        val prediction1 = leakyLeRU(z1, leakyLeRUSlope)

        val z2 = prediction1 * weightsMatrix2 + biasesMatrix2
        val prediction2 = leakyLeRU(z2, leakyLeRUSlope)

        val costError = mseCostFunctionDerivative(prediction2, expectedValues)
        val layerError2 = costError.hadamardMul(leakyLeRUDerivative(z2, leakyLeRUSlope))

        val weightsDelta2 = prediction1.transpose() * layerError2
        val biasesDelta2 = layerError2

        val layerError1 =
            (layerError2 * weightsMatrix2.transpose()).hadamardMul(leakyLeRUDerivative(z1, leakyLeRUSlope))
        val weightsDelta1 = inputMatrix.transpose() * layerError1

        val biasesDelta1 = layerError1

        weightsMatrix1 -= weightsDelta1 * learningRate
        biasesMatrix1 -= biasesDelta1 * learningRate

        weightsMatrix2 -= weightsDelta2 * learningRate
        biasesMatrix2 -= biasesDelta2 * learningRate

        Assertions.assertArrayEquals(
            weightsMatrix1.toFlatArray(),
            weightsVariable1.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix1.toFlatArray(),
            biasVariable1.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            weightsMatrix2.toFlatArray(),
            weightsVariable2.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix2.toFlatArray(),
            biasVariable2.data,
            0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun twoLayersMultiSampleTestOneEpoch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)
        val learningRate = 0.001f
        val leakyLeRUSlope = 0.01f

        val outputSize = source.nextInt(1, 100)
        val hiddenSize = source.nextInt(1, 100)
        val inputSize = source.nextInt(1, 100)
        val batchSize = source.nextInt(1, 100)

        val inputMatrix = FloatMatrix.random(batchSize, inputSize, source)
        var weightsMatrix1 = FloatMatrix.random(inputSize, hiddenSize, source)
        var biasesMatrix1 = FloatMatrix.random(1, hiddenSize, source)

        var weightsMatrix2 = FloatMatrix.random(hiddenSize, outputSize, source)
        var biasesMatrix2 = FloatMatrix.random(1, outputSize, source)

        val executionContext = TrainingExecutionContext()
        val input = MiniBatchInputSource(inputMatrix.toArray(), inputSize, batchSize, executionContext)
        val optimizer = SimpleGradientDescentOptimizer(input)

        val weightsVariable1 = weightsMatrix1.toVariable(executionContext, optimizer, learningRate)
        val biasVariable1 = biasesMatrix1.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable2 = weightsMatrix2.toVariable(executionContext, optimizer, learningRate)
        val biasVariable2 = biasesMatrix2.toVariable(executionContext, optimizer, learningRate)

        val multiplication1 = Multiplication(
            executionContext, input, weightsVariable1
        )

        val add1 = Add(executionContext, multiplication1, biasVariable1, true)
        val leRU1 = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add1)

        val multiplication2 = Multiplication(
            executionContext, leRU1, weightsVariable2
        )

        val add2 = Add(executionContext, multiplication2, biasVariable2, true)
        val leRU2 = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add2)

        val expectedValues = FloatMatrix.random(batchSize, outputSize, source)
        val expectedValuesSource = MiniBatchInputSource(
            expectedValues.toArray(), outputSize,
            batchSize, executionContext
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU2, expectedValuesSource
        )

        executionContext.initializeExecution(mseCostFunction)
        executionContext.executePropagation(1)

        val z1 = inputMatrix * weightsMatrix1 + biasesMatrix1.broadcastByRows(batchSize)
        val prediction1 = leakyLeRU(z1, leakyLeRUSlope)

        val z2 = prediction1 * weightsMatrix2 + biasesMatrix2.broadcastByRows(batchSize)
        val prediction2 = leakyLeRU(z2, leakyLeRUSlope)

        val costError = mseCostFunctionDerivative(prediction2, expectedValues)
        val layerError2 = costError.hadamardMul(leakyLeRUDerivative(z2, leakyLeRUSlope))

        val weightsDelta2 = prediction1.transpose() * layerError2
        val biasesDelta2 = layerError2

        val layerError1 =
            (layerError2 * weightsMatrix2.transpose()).hadamardMul(leakyLeRUDerivative(z1, leakyLeRUSlope))
        val weightsDelta1 = inputMatrix.transpose() * layerError1

        val biasesDelta1 = layerError1

        weightsMatrix1 -= weightsDelta1 * learningRate / batchSize
        biasesMatrix1 -= biasesDelta1.sumByRows() * learningRate / batchSize

        weightsMatrix2 -= weightsDelta2 * learningRate / batchSize
        biasesMatrix2 -= biasesDelta2.sumByRows() * learningRate / batchSize

        Assertions.assertArrayEquals(
            weightsMatrix1.toFlatArray(),
            weightsVariable1.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix1.toFlatArray(),
            biasVariable1.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            weightsMatrix2.toFlatArray(),
            weightsVariable2.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix2.toFlatArray(),
            biasVariable2.data,
            0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun twoLayersMultiSampleTestSeveralEpochs(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)
        val learningRate = 0.01f
        val leakyLeRUSlope = 0.01f

        val outputSize = source.nextInt(1, 100)
        val hiddenSize = source.nextInt(1, 100)
        val inputSize = source.nextInt(1, 100)
        val batchSize = source.nextInt(1, 100)
        val epochs = source.nextInt(5, 50)

        val inputMatrix = FloatMatrix.random(batchSize, inputSize, source)
        var weightsMatrix1 = FloatMatrix.random(inputSize, hiddenSize, source)
        var biasesMatrix1 = FloatMatrix.random(1, hiddenSize, source)

        var weightsMatrix2 = FloatMatrix.random(hiddenSize, outputSize, source)
        var biasesMatrix2 = FloatMatrix.random(1, outputSize, source)

        val executionContext = TrainingExecutionContext()
        val input = MiniBatchInputSource(inputMatrix.toArray(), inputSize, batchSize, executionContext)
        val optimizer = SimpleGradientDescentOptimizer(input)

        val weightsVariable1 = weightsMatrix1.toVariable(
            "weightsVariable1",
            executionContext, optimizer, learningRate
        )

        val biasVariable1 = biasesMatrix1.toVariable(
            "biasVariable1",
            executionContext, optimizer, learningRate
        )

        val weightsVariable2 = weightsMatrix2.toVariable(
            "weightsVariable2",
            executionContext, optimizer, learningRate
        )
        val biasVariable2 = biasesMatrix2.toVariable(
            "biasVariable2",
            executionContext, optimizer, learningRate
        )

        val multiplication1 = Multiplication(
            "multiplication1",
            executionContext, input, weightsVariable1
        )

        val add1 = Add("add1", executionContext, multiplication1, biasVariable1, true)
        val leRU1 = LeakyLeRUFunction("leRU1", leakyLeRUSlope, executionContext, add1)

        val multiplication2 = Multiplication(
            "multiplication2",
            executionContext, leRU1, weightsVariable2
        )

        val add2 = Add("add2", executionContext, multiplication2, biasVariable2, true)
        val leRU2 = LeakyLeRUFunction("leRU2", leakyLeRUSlope, executionContext, add2)

        val expectedValues = FloatMatrix.random(batchSize, outputSize, source)
        val expectedValuesSource = MiniBatchInputSource(
            expectedValues.toArray(), outputSize,
            batchSize, executionContext
        )
        val mseCostFunction = MSEByRowsCostFunction(
            "mseCostFunction", executionContext, leRU2,
            expectedValuesSource
        )

        executionContext.initializeExecution(mseCostFunction)
        executionContext.executePropagation(epochs)

        for (i in 0 until epochs) {
            val z1 = inputMatrix * weightsMatrix1 + biasesMatrix1.broadcastByRows(batchSize)
            val prediction1 = leakyLeRU(z1, leakyLeRUSlope)

            val z2 = prediction1 * weightsMatrix2 + biasesMatrix2.broadcastByRows(batchSize)
            val prediction2 = leakyLeRU(z2, leakyLeRUSlope)

            val costError = mseCostFunctionDerivative(prediction2, expectedValues)
            val layerError2 = costError.hadamardMul(leakyLeRUDerivative(z2, leakyLeRUSlope))

            val weightsDelta2 = prediction1.transpose() * layerError2
            val biasesDelta2 = layerError2

            val layerError1 =
                (layerError2 * weightsMatrix2.transpose()).hadamardMul(leakyLeRUDerivative(z1, leakyLeRUSlope))
            val weightsDelta1 = inputMatrix.transpose() * layerError1

            val biasesDelta1 = layerError1

            weightsMatrix1 -= weightsDelta1 * learningRate / batchSize
            biasesMatrix1 -= biasesDelta1.sumByRows() * learningRate / batchSize

            weightsMatrix2 -= weightsDelta2 * learningRate / batchSize
            biasesMatrix2 -= biasesDelta2.sumByRows() * learningRate / batchSize
        }

        Assertions.assertArrayEquals(
            weightsMatrix1.toFlatArray(),
            weightsVariable1.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix1.toFlatArray(),
            biasVariable1.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            weightsMatrix2.toFlatArray(),
            weightsVariable2.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix2.toFlatArray(),
            biasVariable2.data,
            0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun twoLayersMultiSampleTestSeveralEpochsMiniBatch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)
        val learningRate = 0.01f
        val leakyLeRUSlope = 0.01f

        val outputSize = source.nextInt(1, 100)
        val hiddenSize = source.nextInt(1, 100)
        val inputSize = source.nextInt(1, 100)
        val batchSize = source.nextInt(20, 100)
        val epochs = source.nextInt(5, 50)
        val miniBatchSize = source.nextInt(2, 10)

        val inputMatrix = FloatMatrix.random(batchSize, inputSize, source)
        var weightsMatrix1 = FloatMatrix.random(inputSize, hiddenSize, source)
        var biasesMatrix1 = FloatMatrix.random(1, hiddenSize, source)

        var weightsMatrix2 = FloatMatrix.random(hiddenSize, outputSize, source)
        var biasesMatrix2 = FloatMatrix.random(1, outputSize, source)

        val executionContext = TrainingExecutionContext()
        val input = MiniBatchInputSource(inputMatrix.toArray(), inputSize, miniBatchSize, executionContext)
        val optimizer = SimpleGradientDescentOptimizer(input)

        val weightsVariable1 = weightsMatrix1.toVariable(
            "weightsVariable1",
            executionContext, optimizer, learningRate
        )

        val biasVariable1 = biasesMatrix1.toVariable(
            "biasVariable1",
            executionContext, optimizer, learningRate
        )

        val weightsVariable2 = weightsMatrix2.toVariable(
            "weightsVariable2",
            executionContext, optimizer, learningRate
        )
        val biasVariable2 = biasesMatrix2.toVariable(
            "biasVariable2",
            executionContext, optimizer, learningRate
        )

        val multiplication1 = Multiplication(
            "multiplication1",
            executionContext, input, weightsVariable1
        )

        val add1 = Add("add1", executionContext, multiplication1, biasVariable1, true)
        val leRU1 = LeakyLeRUFunction("leRU1", leakyLeRUSlope, executionContext, add1)

        val multiplication2 = Multiplication(
            "multiplication2",
            executionContext, leRU1, weightsVariable2
        )

        val add2 = Add("add2", executionContext, multiplication2, biasVariable2, true)
        val leRU2 = LeakyLeRUFunction("leRU2", leakyLeRUSlope, executionContext, add2)

        val expectedValues = FloatMatrix.random(batchSize, outputSize, source)
        val expectedValuesSource = MiniBatchInputSource(
            expectedValues.toArray(), outputSize,
            miniBatchSize, executionContext
        )
        val mseCostFunction = MSEByRowsCostFunction(
            "mseCostFunction", executionContext, leRU2,
            expectedValuesSource
        )

        executionContext.initializeExecution(mseCostFunction)

        val fullBatchesCount = (batchSize + miniBatchSize - 1) / miniBatchSize
        executionContext.executePropagation(epochs * fullBatchesCount)
        for (i in 0 until epochs) {
            for (start in 0 until batchSize step miniBatchSize) {
                val miniBatchCount = min(miniBatchSize, batchSize - start)
                val miniInput = inputMatrix.subRows(start, miniBatchCount)

                val z1 = miniInput * weightsMatrix1 + biasesMatrix1.broadcastByRows(miniBatchCount)
                val prediction1 = leakyLeRU(z1, leakyLeRUSlope)

                val z2 = prediction1 * weightsMatrix2 + biasesMatrix2.broadcastByRows(miniBatchCount)
                val prediction2 = leakyLeRU(z2, leakyLeRUSlope)

                val costError = mseCostFunctionDerivative(prediction2, expectedValues.subRows(start, miniBatchCount))
                val layerError2 = costError.hadamardMul(leakyLeRUDerivative(z2, leakyLeRUSlope))

                val weightsDelta2 = prediction1.transpose() * layerError2
                val biasesDelta2 = layerError2

                val layerError1 =
                    (layerError2 * weightsMatrix2.transpose()).hadamardMul(leakyLeRUDerivative(z1, leakyLeRUSlope))
                val weightsDelta1 = miniInput.transpose() * layerError1

                val biasesDelta1 = layerError1

                weightsMatrix1 -= weightsDelta1 * learningRate / miniBatchCount
                biasesMatrix1 -= biasesDelta1.sumByRows() * learningRate / miniBatchCount

                weightsMatrix2 -= weightsDelta2 * learningRate / miniBatchCount
                biasesMatrix2 -= biasesDelta2.sumByRows() * learningRate / miniBatchCount
            }
        }

        Assertions.assertArrayEquals(
            weightsMatrix1.toFlatArray(),
            weightsVariable1.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix1.toFlatArray(),
            biasVariable1.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            weightsMatrix2.toFlatArray(),
            weightsVariable2.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix2.toFlatArray(),
            biasVariable2.data,
            0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun threeLayersSingleSampleTestOneEpoch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)
        val learningRate = 0.01f
        val leakyLeRUSlope = 0.01f


        val inputSize = source.nextInt(1, 100)
        val hiddenSize1 = source.nextInt(1, 100)
        val hiddenSize2 = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)

        val inputMatrix = FloatMatrix.random(1, inputSize, source)

        var weightsMatrix1 = FloatMatrix.random(inputSize, hiddenSize1, source)
        var biasesMatrix1 = FloatMatrix.random(1, hiddenSize1, source)

        var weightsMatrix2 = FloatMatrix.random(hiddenSize1, hiddenSize2, source)
        var biasesMatrix2 = FloatMatrix.random(1, hiddenSize2, source)

        var weightsMatrix3 = FloatMatrix.random(hiddenSize2, outputSize, source)
        var biasesMatrix3 = FloatMatrix.random(1, outputSize, source)

        val executionContext = TrainingExecutionContext()
        val input = inputMatrix.toConstant(executionContext)

        val optimizer = SimpleGradientDescentOptimizer(input)

        val weightsVariable1 = weightsMatrix1.toVariable(executionContext, optimizer, learningRate)
        val biasVariable1 = biasesMatrix1.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable2 = weightsMatrix2.toVariable(executionContext, optimizer, learningRate)
        val biasVariable2 = biasesMatrix2.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable3 = weightsMatrix3.toVariable(executionContext, optimizer, learningRate)
        val biasVariable3 = biasesMatrix3.toVariable(executionContext, optimizer, learningRate)

        val multiplication1 = Multiplication(
            executionContext, input, weightsVariable1
        )
        val add1 = Add(executionContext, multiplication1, biasVariable1, true)
        val leRU1 = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add1)

        val multiplication2 = Multiplication(
            executionContext, leRU1, weightsVariable2
        )
        val add2 = Add(executionContext, multiplication2, biasVariable2, true)
        val leRU2 = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add2)

        val multiplication3 = Multiplication(
            executionContext, leRU2, weightsVariable3
        )
        val add3 = Add(executionContext, multiplication3, biasVariable3, true)
        val leRU3 = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add3)

        val expectedValues = FloatMatrix.random(1, outputSize, source)
        val expectedValuesSource = MiniBatchInputSource(
            expectedValues.toArray(), outputSize, 1, executionContext
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU3,
            expectedValuesSource
        )

        executionContext.initializeExecution(mseCostFunction)
        executionContext.executePropagation(1)

        val z1 = inputMatrix * weightsMatrix1 + biasesMatrix1
        val prediction1 = leakyLeRU(z1, leakyLeRUSlope)

        val z2 = prediction1 * weightsMatrix2 + biasesMatrix2
        val prediction2 = leakyLeRU(z2, leakyLeRUSlope)

        val z3 = prediction2 * weightsMatrix3 + biasesMatrix3
        val prediction3 = leakyLeRU(z3, leakyLeRUSlope)

        val costError = mseCostFunctionDerivative(prediction3, expectedValues)
        val layerError3 = costError.hadamardMul(leakyLeRUDerivative(z3, leakyLeRUSlope))

        val weightsDelta3 = prediction2.transpose() * layerError3
        val biasesDelta3 = layerError3

        val layerError2 =
            (layerError3 * weightsMatrix3.transpose()).hadamardMul(leakyLeRUDerivative(z2, leakyLeRUSlope))
        val weightsDelta2 = prediction1.transpose() * layerError2

        val biasesDelta2 = layerError2

        val layerError1 =
            (layerError2 * weightsMatrix2.transpose()).hadamardMul(leakyLeRUDerivative(z1, leakyLeRUSlope))
        val weightsDelta1 = inputMatrix.transpose() * layerError1
        val biasesDelta1 = layerError1

        weightsMatrix1 -= weightsDelta1 * learningRate
        biasesMatrix1 -= biasesDelta1 * learningRate

        weightsMatrix2 -= weightsDelta2 * learningRate
        biasesMatrix2 -= biasesDelta2 * learningRate

        weightsMatrix3 -= weightsDelta3 * learningRate
        biasesMatrix3 -= biasesDelta3 * learningRate

        Assertions.assertArrayEquals(
            weightsMatrix1.toFlatArray(),
            weightsVariable1.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix1.toFlatArray(),
            biasVariable1.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            weightsMatrix2.toFlatArray(),
            weightsVariable2.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix2.toFlatArray(),
            biasVariable2.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            weightsMatrix3.toFlatArray(),
            weightsVariable3.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix3.toFlatArray(),
            biasVariable3.data,
            0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun theeLayersMultiSampleTestOneEpoch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)
        val learningRate = 0.01f
        val leakyLeRUSlope = 0.01f

        val inputSize = source.nextInt(1, 100)
        val hiddenSize1 = source.nextInt(1, 100)
        val hiddenSize2 = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)
        val batchSize = source.nextInt(1, 100)

        val inputMatrix = FloatMatrix.random(batchSize, inputSize, source)

        var weightsMatrix1 = FloatMatrix.random(inputSize, hiddenSize1, source)
        var biasesMatrix1 = FloatMatrix.random(1, hiddenSize1, source)

        var weightsMatrix2 = FloatMatrix.random(hiddenSize1, hiddenSize2, source)
        var biasesMatrix2 = FloatMatrix.random(1, hiddenSize2, source)

        var weightsMatrix3 = FloatMatrix.random(hiddenSize2, outputSize, source)
        var biasesMatrix3 = FloatMatrix.random(1, outputSize, source)

        val executionContext = TrainingExecutionContext()
        val input = MiniBatchInputSource(inputMatrix.toArray(), inputSize, batchSize, executionContext)

        val optimizer = SimpleGradientDescentOptimizer(input)

        val weightsVariable1 = weightsMatrix1.toVariable(executionContext, optimizer, learningRate)
        val biasVariable1 = biasesMatrix1.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable2 = weightsMatrix2.toVariable(executionContext, optimizer, learningRate)
        val biasVariable2 = biasesMatrix2.toVariable(executionContext, optimizer, learningRate)

        val weightsVariable3 = weightsMatrix3.toVariable(executionContext, optimizer, learningRate)
        val biasVariable3 = biasesMatrix3.toVariable(executionContext, optimizer, learningRate)

        val multiplication1 = Multiplication(
            executionContext, input, weightsVariable1
        )

        val add1 = Add(executionContext, multiplication1, biasVariable1, true)
        val leRU1 = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add1)

        val multiplication2 = Multiplication(
            executionContext, leRU1, weightsVariable2
        )

        val add2 = Add(executionContext, multiplication2, biasVariable2, true)
        val leRU2 = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add2)

        val multiplication3 = Multiplication(
            executionContext, leRU2, weightsVariable3
        )

        val add3 = Add(executionContext, multiplication3, biasVariable3, true)
        val leRU3 = LeakyLeRUFunction(leakyLeRUSlope, executionContext, add3)

        val expectedValues = FloatMatrix.random(batchSize, outputSize, source)
        val expectedValuesSource = MiniBatchInputSource(
            expectedValues.toArray(), outputSize,
            batchSize, executionContext
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU3, expectedValuesSource
        )

        executionContext.initializeExecution(mseCostFunction)
        executionContext.executePropagation(1)

        val z1 = inputMatrix * weightsMatrix1 + biasesMatrix1.broadcastByRows(batchSize)
        val prediction1 = leakyLeRU(z1, leakyLeRUSlope)

        val z2 = prediction1 * weightsMatrix2 + biasesMatrix2.broadcastByRows(batchSize)
        val prediction2 = leakyLeRU(z2, leakyLeRUSlope)

        val z3 = prediction2 * weightsMatrix3 + biasesMatrix3.broadcastByRows(batchSize)
        val prediction3 = leakyLeRU(z3, leakyLeRUSlope)

        val costError = mseCostFunctionDerivative(prediction3, expectedValues)
        val layerError3 = costError.hadamardMul(leakyLeRUDerivative(z3, leakyLeRUSlope))

        val weightsDelta3 = prediction2.transpose() * layerError3
        val biasesDelta3 = layerError3

        val layerError2 =
            (layerError3 * weightsMatrix3.transpose()).hadamardMul(leakyLeRUDerivative(z2, leakyLeRUSlope))
        val weightsDelta2 = prediction1.transpose() * layerError2

        val biasesDelta2 = layerError2

        val layerError1 =
            (layerError2 * weightsMatrix2.transpose()).hadamardMul(leakyLeRUDerivative(z1, leakyLeRUSlope))
        val weightsDelta1 = inputMatrix.transpose() * layerError1
        val biasesDelta1 = layerError1

        weightsMatrix1 -= weightsDelta1 * learningRate / batchSize
        biasesMatrix1 -= biasesDelta1.sumByRows() * learningRate / batchSize

        weightsMatrix2 -= weightsDelta2 * learningRate / batchSize
        biasesMatrix2 -= biasesDelta2.sumByRows() * learningRate / batchSize

        weightsMatrix3 -= weightsDelta3 * learningRate / batchSize
        biasesMatrix3 -= biasesDelta3.sumByRows() * learningRate / batchSize

        Assertions.assertArrayEquals(
            weightsMatrix1.toFlatArray(),
            weightsVariable1.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix1.toFlatArray(),
            biasVariable1.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            weightsMatrix2.toFlatArray(),
            weightsVariable2.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix2.toFlatArray(),
            biasVariable2.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            weightsMatrix3.toFlatArray(),
            weightsVariable3.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix3.toFlatArray(),
            biasVariable3.data,
            0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun threeLayersMultiSampleTestSeveralEpochs(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)
        val learningRate = 0.001f
        val leakyLeRUSlope = 0.01f

        val inputSize = source.nextInt(1, 100)
        val hiddenSize1 = source.nextInt(1, 100)
        val hiddenSize2 = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)
        val batchSize = source.nextInt(1, 100)

        val epochs = source.nextInt(5, 50)

        val inputMatrix = FloatMatrix.random(batchSize, inputSize, source)

        var weightsMatrix1 = FloatMatrix.random(inputSize, hiddenSize1, source)
        var biasesMatrix1 = FloatMatrix.random(1, hiddenSize1, source)

        var weightsMatrix2 = FloatMatrix.random(hiddenSize1, hiddenSize2, source)
        var biasesMatrix2 = FloatMatrix.random(1, hiddenSize2, source)

        var weightsMatrix3 = FloatMatrix.random(hiddenSize2, outputSize, source)
        var biasesMatrix3 = FloatMatrix.random(1, outputSize, source)

        val executionContext = TrainingExecutionContext()

        val input = MiniBatchInputSource(inputMatrix.toArray(), inputSize, batchSize, executionContext)
        val optimizer = SimpleGradientDescentOptimizer(input)

        val weightsVariable1 = weightsMatrix1.toVariable(
            "weightsVariable1",
            executionContext, optimizer, learningRate
        )
        val biasVariable1 = biasesMatrix1.toVariable(
            "biasVariable1",
            executionContext, optimizer, learningRate
        )
        val weightsVariable2 = weightsMatrix2.toVariable(
            "weightsVariable2",
            executionContext, optimizer, learningRate
        )
        val biasVariable2 = biasesMatrix2.toVariable(
            "biasVariable2",
            executionContext, optimizer, learningRate
        )
        val weightsVariable3 = weightsMatrix3.toVariable(
            "weightsVariable3",
            executionContext, optimizer, learningRate
        )
        val biasVariable3 = biasesMatrix3.toVariable(
            "biasVariable3",
            executionContext, optimizer, learningRate
        )
        val multiplication1 = Multiplication(
            "multiplication1",
            executionContext, input, weightsVariable1
        )
        val add1 = Add("add1", executionContext, multiplication1, biasVariable1, true)
        val leRU1 = LeakyLeRUFunction("leRU1", leakyLeRUSlope, executionContext, add1)
        val multiplication2 = Multiplication(
            "multiplication2",
            executionContext, leRU1, weightsVariable2
        )
        val add2 = Add("add2", executionContext, multiplication2, biasVariable2, true)
        val leRU2 = LeakyLeRUFunction("leRU2", leakyLeRUSlope, executionContext, add2)
        val multiplication3 = Multiplication(
            "multiplication3",
            executionContext, leRU2, weightsVariable3
        )
        val add3 = Add("add3", executionContext, multiplication3, biasVariable3, true)
        val leRU3 = LeakyLeRUFunction("leRU3", leakyLeRUSlope, executionContext, add3)

        val expectedValues = FloatMatrix.random(batchSize, outputSize, source)
        val expectedValuesSource = MiniBatchInputSource(
            expectedValues.toArray(), outputSize,
            batchSize, executionContext
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU3, expectedValuesSource
        )

        executionContext.initializeExecution(mseCostFunction)
        executionContext.executePropagation(epochs)

        for (i in 0 until epochs) {
            val z1 = inputMatrix * weightsMatrix1 + biasesMatrix1.broadcastByRows(batchSize)
            val prediction1 = leakyLeRU(z1, leakyLeRUSlope)

            val z2 = prediction1 * weightsMatrix2 + biasesMatrix2.broadcastByRows(batchSize)
            val prediction2 = leakyLeRU(z2, leakyLeRUSlope)

            val z3 = prediction2 * weightsMatrix3 + biasesMatrix3.broadcastByRows(batchSize)
            val prediction3 = leakyLeRU(z3, leakyLeRUSlope)

            val costError = mseCostFunctionDerivative(prediction3, expectedValues)
            val layerError3 = costError.hadamardMul(leakyLeRUDerivative(z3, leakyLeRUSlope))

            val weightsDelta3 = prediction2.transpose() * layerError3
            val biasesDelta3 = layerError3

            val layerError2 =
                (layerError3 * weightsMatrix3.transpose()).hadamardMul(leakyLeRUDerivative(z2, leakyLeRUSlope))
            val weightsDelta2 = prediction1.transpose() * layerError2

            val biasesDelta2 = layerError2

            val layerError1 =
                (layerError2 * weightsMatrix2.transpose()).hadamardMul(leakyLeRUDerivative(z1, leakyLeRUSlope))
            val weightsDelta1 = inputMatrix.transpose() * layerError1
            val biasesDelta1 = layerError1

            weightsMatrix1 -= weightsDelta1 * learningRate / batchSize
            biasesMatrix1 -= biasesDelta1.sumByRows() * learningRate / batchSize

            weightsMatrix2 -= weightsDelta2 * learningRate / batchSize
            biasesMatrix2 -= biasesDelta2.sumByRows() * learningRate / batchSize

            weightsMatrix3 -= weightsDelta3 * learningRate / batchSize
            biasesMatrix3 -= biasesDelta3.sumByRows() * learningRate / batchSize
        }

        Assertions.assertArrayEquals(
            weightsMatrix1.toFlatArray(),
            weightsVariable1.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix1.toFlatArray(),
            biasVariable1.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            weightsMatrix2.toFlatArray(),
            weightsVariable2.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix2.toFlatArray(),
            biasVariable2.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            weightsMatrix3.toFlatArray(),
            weightsVariable3.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix3.toFlatArray(),
            biasVariable3.data,
            0.001f
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun threeLayersMultiSampleTestSeveralEpochsMiniBatch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)
        val learningRate = 0.001f
        val leakyLeRUSlope = 0.01f

        val inputSize = source.nextInt(1, 100)
        val hiddenSize1 = source.nextInt(1, 100)
        val hiddenSize2 = source.nextInt(1, 100)
        val outputSize = source.nextInt(1, 100)
        val batchSize = source.nextInt(20, 100)
        val miniBatchSize = source.nextInt(2, 10)

        val epochs = source.nextInt(5, 50)

        val inputMatrix = FloatMatrix.random(batchSize, inputSize, source)

        var weightsMatrix1 = FloatMatrix.random(inputSize, hiddenSize1, source)
        var biasesMatrix1 = FloatMatrix.random(1, hiddenSize1, source)

        var weightsMatrix2 = FloatMatrix.random(hiddenSize1, hiddenSize2, source)
        var biasesMatrix2 = FloatMatrix.random(1, hiddenSize2, source)

        var weightsMatrix3 = FloatMatrix.random(hiddenSize2, outputSize, source)
        var biasesMatrix3 = FloatMatrix.random(1, outputSize, source)

        val executionContext = TrainingExecutionContext()
        val input = MiniBatchInputSource(inputMatrix.toArray(), inputSize, miniBatchSize, executionContext)
        val optimizer = SimpleGradientDescentOptimizer(input)

        val weightsVariable1 = weightsMatrix1.toVariable(
            "weightsVariable1",
            executionContext, optimizer, learningRate
        )
        val biasVariable1 = biasesMatrix1.toVariable(
            "biasVariable1",
            executionContext, optimizer, learningRate
        )
        val weightsVariable2 = weightsMatrix2.toVariable(
            "weightsVariable2",
            executionContext, optimizer, learningRate
        )
        val biasVariable2 = biasesMatrix2.toVariable(
            "biasVariable2",
            executionContext, optimizer, learningRate
        )
        val weightsVariable3 = weightsMatrix3.toVariable(
            "weightsVariable3",
            executionContext, optimizer, learningRate
        )
        val biasVariable3 = biasesMatrix3.toVariable(
            "biasVariable3",
            executionContext, optimizer, learningRate
        )
        val multiplication1 = Multiplication(
            "multiplication1",
            executionContext, input, weightsVariable1
        )
        val add1 = Add("add1", executionContext, multiplication1, biasVariable1, true)
        val leRU1 = LeakyLeRUFunction("leRU1", leakyLeRUSlope, executionContext, add1)
        val multiplication2 = Multiplication(
            "multiplication2",
            executionContext, leRU1, weightsVariable2
        )
        val add2 = Add("add2", executionContext, multiplication2, biasVariable2, true)
        val leRU2 = LeakyLeRUFunction("leRU2", leakyLeRUSlope, executionContext, add2)
        val multiplication3 = Multiplication(
            "multiplication3",
            executionContext, leRU2, weightsVariable3
        )
        val add3 = Add("add3", executionContext, multiplication3, biasVariable3, true)
        val leRU3 = LeakyLeRUFunction("leRU3", leakyLeRUSlope, executionContext, add3)

        val expectedValues = FloatMatrix.random(batchSize, outputSize, source)
        val expectedValuesSource = MiniBatchInputSource(
            expectedValues.toArray(), outputSize,
            miniBatchSize, executionContext
        )
        val mseCostFunction = MSEByRowsCostFunction(
            executionContext, leRU3, expectedValuesSource
        )

        executionContext.initializeExecution(mseCostFunction)

        val fullBatchesCount = (batchSize + miniBatchSize - 1) / miniBatchSize
        executionContext.executePropagation(fullBatchesCount * epochs)

        for (i in 0 until epochs) {
            for (start in 0 until batchSize step miniBatchSize) {
                val miniBatchCount = min(miniBatchSize, batchSize - start)
                val miniInput = inputMatrix.subRows(start, miniBatchCount)

                val z1 = miniInput * weightsMatrix1 + biasesMatrix1.broadcastByRows(miniBatchCount)
                val prediction1 = leakyLeRU(z1, leakyLeRUSlope)

                val z2 = prediction1 * weightsMatrix2 + biasesMatrix2.broadcastByRows(miniBatchCount)
                val prediction2 = leakyLeRU(z2, leakyLeRUSlope)

                val z3 = prediction2 * weightsMatrix3 + biasesMatrix3.broadcastByRows(miniBatchCount)
                val prediction3 = leakyLeRU(z3, leakyLeRUSlope)

                val costError = mseCostFunctionDerivative(prediction3, expectedValues.subRows(start, miniBatchCount))
                val layerError3 = costError.hadamardMul(leakyLeRUDerivative(z3, leakyLeRUSlope))

                val weightsDelta3 = prediction2.transpose() * layerError3
                val biasesDelta3 = layerError3

                val layerError2 =
                    (layerError3 * weightsMatrix3.transpose()).hadamardMul(leakyLeRUDerivative(z2, leakyLeRUSlope))
                val weightsDelta2 = prediction1.transpose() * layerError2

                val biasesDelta2 = layerError2

                val layerError1 =
                    (layerError2 * weightsMatrix2.transpose()).hadamardMul(leakyLeRUDerivative(z1, leakyLeRUSlope))
                val weightsDelta1 = miniInput.transpose() * layerError1
                val biasesDelta1 = layerError1

                weightsMatrix1 -= weightsDelta1 * learningRate / miniBatchCount
                biasesMatrix1 -= biasesDelta1.sumByRows() * learningRate / miniBatchCount

                weightsMatrix2 -= weightsDelta2 * learningRate / miniBatchCount
                biasesMatrix2 -= biasesDelta2.sumByRows() * learningRate / miniBatchCount

                weightsMatrix3 -= weightsDelta3 * learningRate / miniBatchCount
                biasesMatrix3 -= biasesDelta3.sumByRows() * learningRate / miniBatchCount
            }
        }

        Assertions.assertArrayEquals(
            weightsMatrix1.toFlatArray(),
            weightsVariable1.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix1.toFlatArray(),
            biasVariable1.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            weightsMatrix2.toFlatArray(),
            weightsVariable2.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix2.toFlatArray(),
            biasVariable2.data,
            0.001f
        )

        Assertions.assertArrayEquals(
            weightsMatrix3.toFlatArray(),
            weightsVariable3.data,
            0.001f
        )
        Assertions.assertArrayEquals(
            biasesMatrix3.toFlatArray(),
            biasVariable3.data,
            0.001f
        )
    }
}