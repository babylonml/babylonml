package com.babylonml.backend.training.operations

import com.babylonml.backend.training.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair


class GradientSource(
    executionContext: TrainingExecutionContext, private val rows: Int, private val columns: Int,
    private val gradients: FloatArray, leftOperation: AbstractOperation
) : AbstractOperation(executionContext, leftOperation, null) {
    override fun getResultMaxRows() = rows

    override fun getResultMaxColumns() = columns

    override fun forwardPassCalculation(): Long {
        leftOperation.forwardPassCalculation()
        return TrainingExecutionContext.NULL
    }

    override fun leftBackwardDerivativeChainValue(): Long {
        val result = executionContext.allocateBackwardMemory(rows, columns)
        val resultOffset = TrainingExecutionContext.addressOffset(result)
        val resultBuffer = executionContext.getMemoryBuffer(result)

        System.arraycopy(gradients, 0, resultBuffer, resultOffset, rows * columns)

        return result
    }

    override fun rightBackwardDerivativeChainValue(): Long = TrainingExecutionContext.NULL

    override fun getForwardMemoryAllocations(): Array<IntIntImmutablePair> = emptyArray()

    override fun getBackwardMemoryAllocations(): Array<IntIntImmutablePair> =
        arrayOf(IntIntImmutablePair(rows, columns))

    override fun requiresBackwardDerivativeChainValue(): Boolean = true
}