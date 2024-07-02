package com.babylonml.backend.training.operations

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.cpu.MatrixOperations
import com.babylonml.backend.cpu.VectorOperations
import com.babylonml.backend.training.execution.TensorPointer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import kotlin.math.ln

class SoftMaxCrossEntropyCostFunction(
    expectedProbability: Operation,
    predictedOperation: Operation
) : AbstractOperation(predictedOperation, expectedProbability), CostFunction {
    override val maxResultShape: IntImmutableList =
        CommonTensorOperations.calculateMaxShape(
            predictedOperation.maxResultShape,
            expectedProbability.maxResultShape
        )

    private var softMaxResultPointer: TensorPointer? = null
    private var expectedProbabilityPointer: TensorPointer? = null

    private var trainingMode = false

    override fun forwardPassCalculation(): TensorPointer {
        val predictedOperandResultPointer = leftPreviousOperation!!.forwardPassCalculation()
        val predictedOperandBuffer = predictedOperandResultPointer.buffer()
        val predictedOperandOffset = predictedOperandResultPointer.offset()


        softMaxResultPointer = executionContext.allocateForwardMemory(
            this,
            predictedOperandResultPointer.shape
        )
        expectedProbabilityPointer = rightPreviousOperation!!.forwardPassCalculation()

        val softMaxBuffer = softMaxResultPointer!!.buffer()
        val softMaxOffset = softMaxResultPointer!!.offset()

        val expectedProbability = expectedProbabilityPointer!!.buffer()
        val expectedProbabilityOffset = expectedProbabilityPointer!!.offset()

        val shape = predictedOperandResultPointer.shape
        require(shape.size == 2) { "Softmax cross entropy cost function only supports 2D tensors" }

        MatrixOperations.softMaxByRows(
            predictedOperandBuffer, predictedOperandOffset, shape.getInt(0),
            shape.getInt(1),
            softMaxBuffer, softMaxOffset
        )

        if (trainingMode) {
            return TrainingExecutionContext.NULL
        }

        val stride = CommonTensorOperations.stride(
            predictedOperandResultPointer.shape
        )
        val loopBound = SPECIES.loopBound(stride)
        var vecSum = FloatVector.zero(SPECIES)
        run {
            var i = 0
            while (i < loopBound) {
                val vec = FloatVector.fromArray(
                    SPECIES, softMaxBuffer,
                    softMaxOffset + i
                ).lanewise(VectorOperators.LOG)
                val expectedVec = FloatVector.fromArray(SPECIES, expectedProbability, i + expectedProbabilityOffset)
                vecSum = vec.fma(expectedVec, vecSum)
                i += SPECIES.length()
            }
        }

        var sum = vecSum.reduceLanes(VectorOperators.ADD)
        for (i in loopBound until stride) {
            sum += ln(softMaxBuffer[softMaxOffset + i].toDouble()).toFloat() * expectedProbability[i + expectedProbabilityOffset]
        }

        val result = executionContext.allocateForwardMemory(this, IntImmutableList.of(1, 1))

        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        resultBuffer[resultOffset] = -sum

        return result
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        val softMaxBuffer = softMaxResultPointer!!.buffer()
        val softMaxOffset = softMaxResultPointer!!.offset()

        val expectedProbability = expectedProbabilityPointer!!.buffer()
        val expectedProbabilityOffset = expectedProbabilityPointer!!.offset()

        val result = executionContext.allocateBackwardMemory(this, softMaxResultPointer!!.shape)
        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        val stride = CommonTensorOperations.stride(
            softMaxResultPointer!!.shape
        )
        VectorOperations.subtractVectorFromVector(
            softMaxBuffer, softMaxOffset, expectedProbability,
            expectedProbabilityOffset, resultBuffer, resultOffset, stride
        )

        return result
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        return TrainingExecutionContext.NULL
    }

    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(
            maxResultShape,
            IntImmutableList.of(1, 1)
        )

    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(maxResultShape)

    override fun trainingMode() {
        trainingMode = true
    }

    override fun fullPassCalculationMode() {
        trainingMode = false
    }

    override val requiresBackwardDerivativeChainValue: Boolean by lazy {
        leftPreviousOperation!!.requiresBackwardDerivativeChainValue ||
                rightPreviousOperation!!.requiresBackwardDerivativeChainValue
    }

    companion object {
        private val SPECIES: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED
    }
}
