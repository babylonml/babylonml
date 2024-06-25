package com.babylonml.backend.training.operations

import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies

class MSECostFunction(
    name: String?,
    predictionOperation: Operation,
    expectedValuesOperation: Operation
) : AbstractOperation(name, predictionOperation, expectedValuesOperation), CostFunction {
    override val maxResultShape: IntImmutableList =
        TensorOperations.calculateMaxShape(
            predictionOperation.maxResultShape,
            expectedValuesOperation.maxResultShape
        )

    private var predictionOperandPointer: TensorPointer? = null
    private var expectedValuesPointer: TensorPointer? = null

    private var trainingMode = false

    constructor(
        predictionOperation: Operation,
        expectedValuesOperation: Operation
    ) : this(null, predictionOperation, expectedValuesOperation)


    override fun forwardPassCalculation(): TensorPointer {
        predictionOperandPointer = leftPreviousOperation!!.forwardPassCalculation()
        expectedValuesPointer = rightPreviousOperation!!.forwardPassCalculation()

        if (trainingMode) {
            return TrainingExecutionContext.NULL
        }

        val predictionOperandBuffer = predictionOperandPointer!!.buffer()
        val predictionOperandOffset = predictionOperandPointer!!.offset()

        val expectedValuesBuffer = expectedValuesPointer!!.buffer()
        val expectedValuesOffset = expectedValuesPointer!!.offset()

        val result = executionContext.allocateForwardMemory(this, IntImmutableList.of(1, 1))
        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        val stride = TensorOperations.stride(predictionOperandPointer!!.shape)
        val loopBound = SPECIES.loopBound(stride)

        var vecSum = FloatVector.zero(SPECIES)
        run {
            var i = 0
            while (i < loopBound) {
                val vec = FloatVector.fromArray(SPECIES, predictionOperandBuffer, predictionOperandOffset + i)
                val expectedVec = FloatVector.fromArray(SPECIES, expectedValuesBuffer, i + expectedValuesOffset)

                val diff = vec.sub(expectedVec)
                vecSum = diff.fma(diff, vecSum)
                i += SPECIES.length()
            }
        }

        var sum = vecSum.reduceLanes(VectorOperators.ADD)
        for (i in loopBound until stride) {
            val value = predictionOperandBuffer[predictionOperandOffset + i] -
                    expectedValuesBuffer[i + expectedValuesOffset]
            sum += value * value
        }

        resultBuffer[resultOffset] = sum

        return result
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        val predictionOperandBuffer = predictionOperandPointer!!.buffer()
        val predictionOperandOffset = predictionOperandPointer!!.offset()

        val expectedValuesBuffer = expectedValuesPointer!!.buffer()
        val expectedValuesOffset = expectedValuesPointer!!.offset()

        val result = executionContext.allocateBackwardMemory(this, predictionOperandPointer!!.shape)
        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        val stride = TensorOperations.stride(predictionOperandPointer!!.shape)
        val loopBound = SPECIES.loopBound(stride)

        run {
            var i = 0
            while (i < loopBound) {
                val vec = FloatVector.fromArray(SPECIES, predictionOperandBuffer, predictionOperandOffset + i)
                val expectedVec = FloatVector.fromArray(SPECIES, expectedValuesBuffer, i + expectedValuesOffset)

                val diff = vec.sub(expectedVec)
                diff.intoArray(resultBuffer, resultOffset + i)
                i += SPECIES.length()
            }
        }

        for (i in loopBound until stride) {
            resultBuffer[i + resultOffset] = (predictionOperandBuffer[i + predictionOperandOffset]
                    - expectedValuesBuffer[i + expectedValuesOffset])
        }

        return result
    }

    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(IntImmutableList.of(1, 1))

    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(maxResultShape)

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        return TrainingExecutionContext.NULL
    }

    override val requiresBackwardDerivativeChainValue: Boolean by lazy {
        leftPreviousOperation!!.requiresBackwardDerivativeChainValue
    }

    override fun trainingMode() {
        trainingMode = true
    }

    override fun fullPassCalculationMode() {
        trainingMode = false
    }

    companion object {
        private val SPECIES: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED
    }
}
