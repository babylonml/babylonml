package com.babylonml.backend.training.operations

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.cpu.VectorOperations
import com.babylonml.backend.training.execution.TensorPointer
import it.unimi.dsi.fastutil.ints.IntImmutableList

class HadamardProduct(leftOperation: Operation, rightOperation: Operation) :
    AbstractOperation(leftOperation, rightOperation) {
    private var leftOperandPointer: TensorPointer? = null
    private var rightOperandPointer: TensorPointer? = null

    private val maxShape: IntImmutableList =
        CommonTensorOperations.calculateMaxShape(
            leftOperation.maxResultShape,
            rightOperation.maxResultShape
        )

    override val maxResultShape: IntImmutableList
        get() = maxShape

    override fun forwardPassCalculation(): TensorPointer {
        leftOperandPointer = leftPreviousOperation!!.forwardPassCalculation()
        rightOperandPointer = rightPreviousOperation!!.forwardPassCalculation()

        return broadcastIfNeeded(
            leftOperandPointer!!, rightOperandPointer!!, forwardMemoryAllocator
        ) { firstTensor: TensorPointer, secondTensor: TensorPointer, result: TensorPointer ->
            VectorOperations.vectorToVectorElementWiseMultiplication(
                firstTensor.buffer(), firstTensor.offset(),
                secondTensor.buffer(), secondTensor.offset(), result.buffer(), result.offset(),
                CommonTensorOperations.stride(result.shape)
            )
        }
    }

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        return reduceIfNeeded(
            derivativeChainPointer!!, rightOperandPointer!!, backwardMemoryAllocator
        ) { derivativeTensor: TensorPointer, rightTensor: TensorPointer, resultTensor: TensorPointer ->
            VectorOperations.vectorToVectorElementWiseMultiplication(
                derivativeTensor.buffer(), derivativeTensor.offset(),
                rightTensor.buffer(), rightTensor.offset(),
                resultTensor.buffer(), resultTensor.offset(),
                CommonTensorOperations.stride(resultTensor.shape)
            )
        }
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        return reduceIfNeeded(
            derivativeChainPointer!!, leftOperandPointer!!, backwardMemoryAllocator
        ) { derivativeTensor: TensorPointer, leftTensor: TensorPointer, resultTensor: TensorPointer ->
            VectorOperations.vectorToVectorElementWiseMultiplication(
                derivativeTensor.buffer(), derivativeTensor.offset(),
                leftTensor.buffer(), leftTensor.offset(),
                resultTensor.buffer(), resultTensor.offset(),
                CommonTensorOperations.stride(resultTensor.shape)
            )
        }
    }

    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(maxShape)

    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(maxShape, maxShape)

    override val requiresBackwardDerivativeChainValue: Boolean by lazy {
        leftPreviousOperation!!.requiresBackwardDerivativeChainValue ||
                rightPreviousOperation!!.requiresBackwardDerivativeChainValue
    }
}
