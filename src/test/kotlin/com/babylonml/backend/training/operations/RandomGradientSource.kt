package com.babylonml.backend.training.operations

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList
import org.apache.commons.rng.UniformRandomProvider

class RandomGradientSource(
    executionContext: TrainingExecutionContext, private val shape: IntImmutableList,
    private val source: UniformRandomProvider,
    leftOperation: AbstractOperation
) : AbstractOperation(
    executionContext, leftOperation,
    null
), CostFunction {
    val generatedGradients = mutableListOf<FloatArray>()

    override fun forwardPassCalculation(): TensorPointer {
        return leftPreviousOperation!!.forwardPassCalculation()
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        val result = executionContext.allocateBackwardMemory(this, shape)
        val resultOffset = result.offset()
        val resultBuffer = result.buffer()

        val stride = CommonTensorOperations.stride(shape)
        val gradients = FloatArray(stride)
        for (i in 0 until stride) {
            gradients[i] = source.nextFloat(-1.0f, 1.0f)
        }

        System.arraycopy(gradients, 0, resultBuffer, resultOffset, stride)
        generatedGradients.add(gradients)

        return result
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer = TrainingExecutionContext.NULL

    override fun trainingMode() {
        //No-op
    }

    override fun fullPassCalculationMode() {
        //No-op
    }

    override val maxResultShape: IntImmutableList
        get() = shape
    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = emptyList()
    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(shape)
    override val requiresBackwardDerivativeChainValue: Boolean
        get() = true
}