package com.babylonml.backend.training.operations

import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import it.unimi.dsi.fastutil.ints.IntImmutableList

class GradientSource(
    private val shape: IntImmutableList,
    private val gradients: FloatArray, leftOperation: AbstractOperation
) : AbstractOperation(leftOperation, null), CostFunction {
    override fun getMaxResultShape(): IntImmutableList = shape

    override fun forwardPassCalculation(): TensorPointer {
        return leftOperation!!.forwardPassCalculation()
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        val result = executionContext.allocateBackwardMemory(this, shape)
        val resultOffset = result.offset()
        val resultBuffer = result.buffer()

        System.arraycopy(gradients, 0, resultBuffer, resultOffset, TensorOperations.stride(shape))

        return result
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        val result = executionContext.allocateBackwardMemory(this, shape)
        val resultOffset = result.offset()
        val resultBuffer = result.buffer()

        System.arraycopy(gradients, 0, resultBuffer, resultOffset, TensorOperations.stride(shape))

        return result
    }

    override fun getForwardMemoryAllocations(): List<IntImmutableList> = emptyList()

    override fun getBackwardMemoryAllocations(): List<IntImmutableList> =  listOf(shape)

    override fun requiresBackwardDerivativeChainValue(): Boolean = true

    override fun trainingMode() {
        //No-op
    }

    override fun fullPassCalculationMode() {
        //No-op
    }
}