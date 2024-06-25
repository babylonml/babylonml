package com.babylonml.backend.training.operations

import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList


class ResultMemoryCellCostFunction(operation: Operation) : AbstractOperation(operation, null), CostFunction {
    var result = FloatArray(0)
    private val maxShape = operation.maxResultShape

    private lateinit var shape: IntImmutableList

    override fun forwardPassCalculation(): TensorPointer {
        val resultPointer = leftPreviousOperation!!.forwardPassCalculation()

        val resultOffset = resultPointer.offset()
        val resultBuffer = resultPointer.buffer()

        shape = resultPointer.shape()

        val size = TensorOperations.stride(shape)

        result = FloatArray(size)
        System.arraycopy(resultBuffer, resultOffset, result, 0, size)

        return resultPointer
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        val diffResult = executionContext.allocateBackwardMemory(this, shape)

        val diffResultOffset = diffResult.offset()
        val diffResultBuffer = diffResult.buffer()

        val stride = TensorOperations.stride(shape)
        diffResultBuffer.fill(0f, diffResultOffset, diffResultOffset + stride)
        return diffResult
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer = TrainingExecutionContext.NULL

    override fun trainingMode() {
        //No-op
    }

    override fun fullPassCalculationMode() {
        //No-op
    }

    override val maxResultShape: IntImmutableList
        get() = maxShape
    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = emptyList()
    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(maxShape)
    override val requiresBackwardDerivativeChainValue: Boolean
        get() = true
}
