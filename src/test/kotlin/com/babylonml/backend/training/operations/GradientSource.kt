package com.babylonml.backend.training.operations

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import it.unimi.dsi.fastutil.ints.IntImmutableList

class GradientSource(
    private val gradient: Tensor, leftOperation: AbstractOperation
) : AbstractOperation(leftOperation, null), CostFunction {
    override fun forwardPassCalculation(): TensorPointer {
        return leftPreviousOperation!!.forwardPassCalculation()
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        val result = executionContext.allocateBackwardMemory(this, gradient.shape)
        val resultOffset = result.offset()
        val resultBuffer = result.buffer()

        System.arraycopy(gradient.data, 0, resultBuffer, resultOffset,
            CommonTensorOperations.stride(gradient.shape)
        )

        return result
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        val result = executionContext.allocateBackwardMemory(this, gradient.shape)
        val resultOffset = result.offset()
        val resultBuffer = result.buffer()

        System.arraycopy(gradient, 0, resultBuffer, resultOffset,
            CommonTensorOperations.stride(gradient.shape)
        )

        return result
    }

    override fun trainingMode() {
        //No-op
    }

    override fun fullPassCalculationMode() {
        //No-op
    }

    override val maxResultShape: IntImmutableList
        get() = gradient.shape
    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = emptyList()
    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(gradient.shape)
    override val requiresBackwardDerivativeChainValue: Boolean
        get() = true
}