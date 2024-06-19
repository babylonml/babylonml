package com.babylonml.backend.training.operations

import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.training.execution.TensorPointer

class GradientSource(
    private val shape: IntArray,
    private val gradients: FloatArray, leftOperation: AbstractOperation
) : AbstractOperation(leftOperation, null), CostFunction {
    override fun getMaxResultShape(): IntArray = shape

    override fun forwardPassCalculation(): TensorPointer {
        return leftOperation.forwardPassCalculation()
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        val result = executionContext.allocateBackwardMemory(*shape)
        val resultOffset = result.offset()
        val resultBuffer = result.buffer()

        System.arraycopy(gradients, 0, resultBuffer, resultOffset, TensorOperations.stride(shape))

        return result
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        val result = executionContext.allocateBackwardMemory(*shape)
        val resultOffset = result.offset()
        val resultBuffer = result.buffer()

        System.arraycopy(gradients, 0, resultBuffer, resultOffset, TensorOperations.stride(shape))

        return result
    }

    override fun getForwardMemoryAllocations(): Array<IntArray> = emptyArray()

    override fun getBackwardMemoryAllocations(): Array<IntArray> =
        arrayOf(shape)

    override fun requiresBackwardDerivativeChainValue(): Boolean = true

    override fun trainingMode() {
        //No-op
    }

    override fun fullPassCalculation() {
        //No-op
    }
}