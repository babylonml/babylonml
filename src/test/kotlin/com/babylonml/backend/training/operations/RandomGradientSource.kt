package com.babylonml.backend.training.operations

import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import org.apache.commons.rng.UniformRandomProvider

class RandomGradientSource(
    executionContext: TrainingExecutionContext, private val shape: IntArray,
    private val source: UniformRandomProvider,
    leftOperation: AbstractOperation
) : AbstractOperation(
    executionContext, leftOperation,
    null
), CostFunction {
    val generatedGradients = mutableListOf<FloatArray>()

    override fun getMaxResultShape(): IntArray = shape

    override fun forwardPassCalculation(): TensorPointer {
        return leftOperation!!.forwardPassCalculation()
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        val result = executionContext.allocateBackwardMemory(this, *shape)
        val resultOffset = result.offset()
        val resultBuffer = result.buffer()

        val stride = TensorOperations.stride(shape)
        val gradients = FloatArray(stride)
        for (i in 0 until stride) {
            gradients[i] = source.nextFloat(-1.0f, 1.0f)
        }

        System.arraycopy(gradients, 0, resultBuffer, resultOffset, stride)
        generatedGradients.add(gradients)

        return result
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer = TrainingExecutionContext.NULL

    override fun getForwardMemoryAllocations(): Array<IntArray> = emptyArray()

    override fun getBackwardMemoryAllocations(): Array<IntArray> =
        arrayOf(shape)

    override fun requiresBackwardDerivativeChainValue(): Boolean = true

    override fun trainingMode() {
        //No-op
    }

    override fun fullPassCalculationMode() {
        //No-op
    }

}