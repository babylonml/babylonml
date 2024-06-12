package com.babylonml.backend.training.operations

import com.babylonml.backend.training.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair
import org.apache.commons.rng.UniformRandomProvider

class RandomGradientSource(
    executionContext: TrainingExecutionContext, private val rows: Int, private val columns: Int,
    private val source: UniformRandomProvider,
    leftOperation: AbstractOperation
) : AbstractOperation(
    executionContext, leftOperation,
    null
) {
    val generatedGradients = mutableListOf<FloatArray>()

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

        val gradients = FloatArray(rows * columns)
        for (i in 0 until rows * columns) {
            gradients[i] = source.nextFloat(-1.0f, 1.0f)
        }

        System.arraycopy(gradients, 0, resultBuffer, resultOffset, rows * columns)
        generatedGradients.add(gradients)

        return result
    }

    override fun rightBackwardDerivativeChainValue(): Long = TrainingExecutionContext.NULL

    override fun getForwardMemoryAllocations(): Array<IntIntImmutablePair> = emptyArray()

    override fun getBackwardMemoryAllocations(): Array<IntIntImmutablePair> =
        arrayOf(IntIntImmutablePair(rows, columns))

    override fun requiresBackwardDerivativeChainValue(): Boolean = true

}