package com.babylonml.backend.training.operations

import com.babylonml.backend.training.TrainingExecutionContext


class GradientSource(
    executionContext: TrainingExecutionContext, private val rows: Int, private val columns: Int,
    private val gradients: FloatArray, leftOperation: AbstractOperation
) : AbstractOperation(executionContext, leftOperation, null) {
    override fun forwardPassCalculation(): Long = TrainingExecutionContext.NULL

    override fun getForwardMemorySize(): Int = 0

    override fun leftBackwardDerivativeChainValue(): Long {
        val result = executionContext.allocateBackwardMemory(rows * columns)
        val resultOffset = TrainingExecutionContext.addressOffset(result)
        val resultBuffer = executionContext.getMemoryBuffer(result)

        System.arraycopy(gradients, 0, resultBuffer, resultOffset, rows * columns)

        return result
    }

    override fun rightBackwardDerivativeChainValue(): Long = TrainingExecutionContext.NULL

    override fun getBackwardMemorySize(): Int = rows * columns

    override fun requiresBackwardDerivativeChainValue(): Boolean = true
}