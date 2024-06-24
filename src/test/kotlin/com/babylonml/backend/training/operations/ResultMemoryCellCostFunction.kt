package com.babylonml.backend.training.operations

import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import com.babylonml.backend.training.execution.TrainingExecutionContext


class ResultMemoryCellCostFunction(operation: Operation) : AbstractOperation(operation, null), CostFunction {
    var result = FloatArray(0)
    private val maxShape = operation.getMaxResultShape()

    private lateinit var shape: IntArray

    override fun getMaxResultShape(): IntArray = maxShape

    override fun forwardPassCalculation(): TensorPointer {
        val resultPointer = leftOperation!!.forwardPassCalculation()

        val resultOffset = resultPointer.offset()
        val resultBuffer = resultPointer.buffer()

        shape = resultPointer.shape()

        val size = TensorOperations.stride(shape)

        result = FloatArray(size)
        System.arraycopy(resultBuffer, resultOffset, result, 0, size)

        return resultPointer
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        val diffResult = executionContext.allocateBackwardMemory(this, *shape)

        val diffResultOffset = diffResult.offset()
        val diffResultBuffer = diffResult.buffer()

        val stride = TensorOperations.stride(shape)
        diffResultBuffer.fill(0f, diffResultOffset, diffResultOffset + stride)
        return diffResult
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer = TrainingExecutionContext.NULL

    override fun getForwardMemoryAllocations(): Array<IntArray> = emptyArray()

    override fun getBackwardMemoryAllocations(): Array<IntArray> = arrayOf(maxShape)

    override fun requiresBackwardDerivativeChainValue(): Boolean = false

    override fun trainingMode() {
        //No-op
    }

    override fun fullPassCalculationMode() {
        //No-op
    }
}
