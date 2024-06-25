package com.babylonml.backend.training.operations

import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies

class LeakyLeRUFunction(
    name: String?, private val leakyLeRUASlope: Float,
    leftOperation: Operation
) : AbstractOperation(name, leftOperation, null) {
    override val maxResultShape: IntImmutableList =
        TensorOperations.calculateMaxShape(
            leftOperation.maxResultShape,
            leftOperation.maxResultShape
        )

    private var leftOperandResult: TensorPointer? = null

    constructor(leakyLeRUASlope: Float, leftOperation: Operation) : this(null, leakyLeRUASlope, leftOperation)

    override fun forwardPassCalculation(): TensorPointer {
        leftOperandResult = leftPreviousOperation!!.forwardPassCalculation()
        val result = executionContext.allocateForwardMemory(this, leftOperandResult!!.shape)

        val leftResultBuffer = leftOperandResult!!.buffer()
        val leftResultOffset = leftOperandResult!!.offset()

        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        val size = TensorOperations.stride(leftOperandResult!!.shape)
        val loopBound = SPECIES.loopBound(size)
        val zero = FloatVector.zero(SPECIES)

        run {
            var i = 0
            while (i < loopBound) {
                val va = FloatVector.fromArray(SPECIES, leftResultBuffer, leftResultOffset + i)
                val mask = va.compare(VectorOperators.LT, zero)
                val vc = va.mul(leakyLeRUASlope, mask)
                vc.intoArray(resultBuffer, resultOffset + i)
                i += SPECIES.length()
            }
        }

        for (i in loopBound until size) {
            val leftValue = leftResultBuffer[leftResultOffset + i]
            resultBuffer[i + resultOffset] = if (leftValue > 0) leftValue else leakyLeRUASlope * leftValue
        }

        return result
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        val leftOperandBuffer = leftOperandResult!!.buffer()
        val leftOperandOffset = leftOperandResult!!.offset()

        val derivativeChainBuffer = derivativeChainPointer!!.buffer()
        val derivativeChainOffset = derivativeChainPointer!!.offset()

        val result = executionContext.allocateBackwardMemory(this, leftOperandResult!!.shape)

        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        val size = TensorOperations.stride(leftOperandResult!!.shape)

        val loopBound = SPECIES.loopBound(size)
        val zero = FloatVector.zero(SPECIES)
        val slope = FloatVector.broadcast(SPECIES, leakyLeRUASlope)
        val one = FloatVector.broadcast(SPECIES, 1.0f)

        run {
            var i = 0
            while (i < loopBound) {
                val va = FloatVector.fromArray(SPECIES, leftOperandBuffer, leftOperandOffset + i)
                val mask = va.compare(VectorOperators.LT, zero)
                var vc = one.mul(slope, mask)

                val diff = FloatVector.fromArray(SPECIES, derivativeChainBuffer, derivativeChainOffset + i)
                vc = vc.mul(diff)

                vc.intoArray(resultBuffer, resultOffset + i)
                i += SPECIES.length()
            }
        }

        for (i in loopBound until size) {
            resultBuffer[i + resultOffset] =
                (if (leftOperandBuffer[i + leftOperandOffset] > 0) 1.0f else leakyLeRUASlope) *
                        derivativeChainBuffer[i + derivativeChainOffset]
        }

        return result
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        return TrainingExecutionContext.NULL
    }

    override val requiresBackwardDerivativeChainValue: Boolean by lazy {
        leftPreviousOperation!!.requiresBackwardDerivativeChainValue
    }

    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(maxResultShape)

    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(maxResultShape)

    companion object {
        private val SPECIES: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED
    }
}
