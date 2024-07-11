package com.babylonml.backend.training.operations

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.util.*
import kotlin.math.sqrt
import kotlin.math.tanh

class GeLUFunction(leftOperation: Operation) : AbstractOperation(leftOperation, null) {
    override val maxResultShape: IntImmutableList = leftOperation.maxResultShape

    private var leftOperandPointer: TensorPointer? = null

    override fun forwardPassCalculation(): TensorPointer {
        leftOperandPointer = leftPreviousOperation!!.forwardPassCalculation()
        val leftOperandBuffer = executionContext.getMemoryBuffer(leftOperandPointer!!.pointer)
        val leftOperandOffset = TrainingExecutionContext.addressOffset(leftOperandPointer!!.pointer)

        val result = executionContext.allocateForwardMemory(this, leftOperandPointer!!.shape)

        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        val stride = CommonTensorOperations.stride(
            leftOperandPointer!!.shape
        )

        val loopBound = SPECIES.loopBound(stride)
        run {
            var i = 0
            while (i < loopBound) {
                val va = FloatVector.fromArray(SPECIES, leftOperandBuffer, leftOperandOffset + i)
                // GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x^3)))
                val vc = FloatVector.broadcast(SPECIES, SCALAR_1).mul(va)
                    .mul( // 1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
                        FloatVector.broadcast(SPECIES, SCALAR_2).add( // tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
                            FloatVector.broadcast(SPECIES, SCALAR_3).mul( //x + 0.044715 * x^3
                                va.add( // 0.044715 * x^3
                                    va.mul(va).mul(va).mul(SCALAR_4)
                                )

                            ).lanewise(VectorOperators.TANH)
                        )
                    )
                vc.intoArray(resultBuffer, resultOffset + i)
                i += SPECIES.length()
            }
        }


        for (i in loopBound until stride) {
            val leftValue = leftOperandBuffer[leftOperandOffset + i]
            // GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x^3)))
            resultBuffer[i + resultOffset] = (leftValue * SCALAR_1 * (SCALAR_2 + tanh(
                (SCALAR_3 *
                        (leftValue + SCALAR_4 * leftValue * leftValue * leftValue)).toDouble()
            ))).toFloat()
        }

        return result
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        val result = executionContext.allocateBackwardMemory(this, derivativeChainPointer!!.shape)

        val derivativeBuffer = derivativeChainPointer!!.buffer()
        val derivativeOffset = derivativeChainPointer!!.offset()

        Objects.requireNonNull(leftOperandPointer)

        val leftBuffer = leftOperandPointer!!.buffer()
        val leftOffset = leftOperandPointer!!.offset()

        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        val size = CommonTensorOperations.stride(
            leftOperandPointer!!.shape
        )

        val loopBound = SPECIES.loopBound(size)
        run {
            var i = 0
            while (i < loopBound) {
                val value = FloatVector.fromArray(SPECIES, leftBuffer, leftOffset + i)

                //h = tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
                val h = FloatVector.broadcast(SPECIES, SCALAR_3).mul(
                    value.add(
                        value.mul(value).mul(value).mul(FloatVector.broadcast(SPECIES, SCALAR_4))
                    )
                ).lanewise(VectorOperators.TANH)
                // d(GeLU(x))/dx = 0.5 * (1 + h + x * (1 - h^2) * (sqrt(2 / PI) + 3 * 0.044715 * x^2))
                var vc = FloatVector.broadcast(SPECIES, SCALAR_1).mul(
                    FloatVector.broadcast(SPECIES, SCALAR_2).add(h)
                        .add( //x * (1 - h^2) * (sqrt(2 / PI) + 3 * 0.044715 * x^2))
                            value.mul( //sqrt(2 / PI) + 3 * 0.044715 * x^2)
                                FloatVector.broadcast(SPECIES, SCALAR_3).add(
                                    value.mul(value).mul(
                                        FloatVector.broadcast(SPECIES, SCALAR_5)
                                    )
                                ).mul( //(1 - h^2)
                                    FloatVector.broadcast(SPECIES, SCALAR_2).sub(h.mul(h))
                                )
                            )
                        )
                )

                val derivative = FloatVector.fromArray(SPECIES, derivativeBuffer, derivativeOffset + i)
                vc = vc.mul(derivative)

                vc.intoArray(resultBuffer, resultOffset + i)
                i += SPECIES.length()
            }
        }

        for (i in loopBound until size) {
            val value = leftBuffer[leftOffset + i]
            //h = tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
            val h = tanh((SCALAR_3 * (value + SCALAR_4 * value * value * value)).toDouble())
                .toFloat()
            // d(GeLU(x))/dx = 0.5 * (1 + h + x * (1 - h^2) * (sqrt(2 / PI) + 3 * 0.044715 * x^2))
            resultBuffer[i + resultOffset] =
                (SCALAR_1 *
                        (SCALAR_2 + h + value * (SCALAR_3 +
                                SCALAR_5 * value * value) * (SCALAR_2 - h * h))) * derivativeBuffer[i
                        + derivativeOffset]
        }

        return result
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        return TrainingExecutionContext.NULL
    }

    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(maxResultShape)

    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(maxResultShape)

    override val requiresBackwardDerivativeChainValue: Boolean by lazy {
        leftPreviousOperation!!.requiresBackwardDerivativeChainValue
    }

    companion object {
        private val SPECIES: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED
        private const val SCALAR_1 = 0.5f
        private const val SCALAR_2 = 1.0f
        private val SCALAR_3 = sqrt(2 / Math.PI).toFloat()
        private const val SCALAR_4 = 0.044715f
        private const val SCALAR_5 = 3 * SCALAR_4
    }
}
