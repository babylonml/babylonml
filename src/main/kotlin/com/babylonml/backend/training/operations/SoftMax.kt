package com.babylonml.backend.training.operations

import com.babylonml.backend.training.execution.TensorPointer
import it.unimi.dsi.fastutil.ints.IntImmutableList


class SoftMax(leftOperation: Operation) : AbstractOperation(leftOperation, null) {
    override val maxResultShape: IntImmutableList =
        leftOperation.maxResultShape

    override fun forwardPassCalculation(): TensorPointer {
        throw UnsupportedOperationException(
            "This is stub class that is used to implement mix of cross entropy" +
                    " and softmax. It should not be used in forward pass"
        )
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        throw UnsupportedOperationException(
            "This is stub class that is used to implement mix of cross entropy" +
                    " and softmax. It should not be used in backward pass"
        )
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        throw UnsupportedOperationException(
            "This is stub class that is used to implement mix of cross entropy" +
                    " and softmax. It should not be used in backward pass"
        )
    }

    override val forwardMemoryAllocations: List<IntImmutableList>
        get() {
            throw UnsupportedOperationException(
                "This is stub class that is used to implement mix of cross entropy" +
                        " and softmax. It should not be used in forward pass"
            )
        }

    override val backwardMemoryAllocations: List<IntImmutableList>
        get() {
            throw UnsupportedOperationException(
                "This is stub class that is used to implement mix of cross entropy" +
                        " and softmax. It should not be used in backward pass"
            )
        }

    override val requiresBackwardDerivativeChainValue: Boolean
        get() = false
}
