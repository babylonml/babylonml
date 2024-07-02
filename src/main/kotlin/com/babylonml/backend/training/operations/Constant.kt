package com.babylonml.backend.training.operations

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList


@Suppress("unused")
class Constant(
    name: String?,
    executionContext: TrainingExecutionContext,
    private val constant: Tensor
) : AbstractOperation(name, executionContext, null, null), StartOperation {
    constructor(executionContext: TrainingExecutionContext, constant: Tensor) : this(
        null,
        executionContext,
        constant
    )

    override val maxResultShape: IntImmutableList
        get() = constant.shape

    override fun forwardPassCalculation(): TensorPointer {
        val result = executionContext.allocateForwardMemory(this, constant.shape)

        val stride =
            CommonTensorOperations.stride(constant.shape)
        System.arraycopy(constant.data, 0, result.buffer(), result.offset(), stride)

        return result
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        return TrainingExecutionContext.NULL
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        return TrainingExecutionContext.NULL
    }

    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(constant.shape)

    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = emptyList()

    override val requiresBackwardDerivativeChainValue: Boolean
        get() = false

    override fun calculateGradientUpdate() {
        // No gradient update required
    }
}
