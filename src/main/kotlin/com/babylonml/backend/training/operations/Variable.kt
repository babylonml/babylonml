package com.babylonml.backend.training.operations

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import com.babylonml.backend.training.initializer.Initializer
import com.babylonml.backend.training.optimizer.GradientOptimizer
import it.unimi.dsi.fastutil.ints.IntImmutableList

class Variable(
    name: String?, executionContext: TrainingExecutionContext,
    private val optimizer: GradientOptimizer,
    private val tensor: Tensor, private val learningRate: Float
) : AbstractOperation(name, executionContext, null, null), StartOperation {
    constructor(
        executionContext: TrainingExecutionContext, optimizer: GradientOptimizer,
        data: Tensor, learningRate: Float
    ) : this(null, executionContext, optimizer, data, learningRate)

    constructor(
        name: String?, executionContext: TrainingExecutionContext,
        optimizer: GradientOptimizer,
        shape: IntArray, learningRate: Float, initializer: Initializer
    ) : this(
        name, executionContext, optimizer, initData(shape, initializer),
        learningRate
    )

    override val maxResultShape: IntImmutableList
        get() = tensor.shape

    override fun forwardPassCalculation(): TensorPointer {
        val data = tensor.data
        val shape = this.tensor.shape

        val result = executionContext.allocateForwardMemory(this, shape)
        val resultBuffer = executionContext.getMemoryBuffer(result.pointer)
        val resultOffset = TrainingExecutionContext.addressOffset(result.pointer)

        val stride = CommonTensorOperations.stride(shape)
        System.arraycopy(data, 0, resultBuffer, resultOffset, stride)
        return result
    }

    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(tensor.shape)

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        return TrainingExecutionContext.NULL
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        return TrainingExecutionContext.NULL
    }

    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = optimizer.calculateRequiredMemoryAllocations(tensor.shape)

    override val requiresBackwardDerivativeChainValue: Boolean
        get() = true

    val data get() = tensor.data

    override fun calculateGradientUpdate() {
        val derivativeBuffer = executionContext.getMemoryBuffer(derivativeChainPointer!!.pointer)
        val derivativeOffset = TrainingExecutionContext.addressOffset(derivativeChainPointer!!.pointer)

        optimizer.optimize(
            executionContext, tensor.data, 0, tensor.shape, derivativeBuffer,
            derivativeOffset, learningRate, this
        )
    }

    companion object {
        private fun initData(shape: IntArray, initializer: Initializer): Tensor {
            val data = FloatArray(TensorOperations.stride(shape))
            initializer.initialize(data, 0, shape)

            return Tensor(data, shape)
        }
    }
}
