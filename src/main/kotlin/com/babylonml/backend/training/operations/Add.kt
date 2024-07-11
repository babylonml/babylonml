package com.babylonml.backend.training.operations

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.cpu.VectorOperations
import com.babylonml.backend.training.execution.TensorPointer
import it.unimi.dsi.fastutil.ints.IntImmutableList

class Add(name: String?, leftOperation: Operation, rightOperation: Operation) :
    AbstractOperation(name, leftOperation, rightOperation) {
    private var leftOperandPointer: TensorPointer? = null
    private var rightOperandPointer: TensorPointer? = null


    private val maxShape: IntImmutableList

    constructor(leftOperation: Operation, rightOperation: Operation) : this(null, leftOperation, rightOperation)

    init {
        val leftMaxShape = leftOperation.maxResultShape
        val rightMaxShape = rightOperation.maxResultShape

        maxShape = CommonTensorOperations.calculateMaxShape(
            leftMaxShape,
            rightMaxShape
        )

    }


    override fun forwardPassCalculation(): TensorPointer {
        leftOperandPointer = leftPreviousOperation!!.forwardPassCalculation()
        rightOperandPointer = rightPreviousOperation!!.forwardPassCalculation()

        return broadcastIfNeeded(
            leftOperandPointer!!, rightOperandPointer!!, forwardMemoryAllocator
        ) { firstTensor: TensorPointer, secondTensor: TensorPointer, result: TensorPointer ->
            VectorOperations.addVectorToVector(
                firstTensor.buffer(), firstTensor.offset(), secondTensor.buffer(),
                secondTensor.offset(), result.buffer(),
                result.offset(), CommonTensorOperations.stride(result.shape)
            )
        }
    }

    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(maxShape)

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        return calculateDerivative(leftOperandPointer!!)
    }

    private fun calculateDerivative(operandPointer: TensorPointer): TensorPointer {
        return reduceIfNeeded(
            operandPointer, derivativeChainPointer!!, backwardMemoryAllocator
        ) { operandTensor: TensorPointer, derivativeTensor: TensorPointer, resultTensor: TensorPointer ->
            System.arraycopy(
                derivativeTensor.buffer(),
                derivativeTensor.offset(),
                resultTensor.buffer(), resultTensor.offset(),
                CommonTensorOperations.stride(resultTensor.shape)
            )
        }
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        return calculateDerivative(rightOperandPointer!!)
    }

    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(maxShape, maxShape)

    override val requiresBackwardDerivativeChainValue: Boolean by lazy {
        leftPreviousOperation!!.requiresBackwardDerivativeChainValue ||
                rightPreviousOperation!!.requiresBackwardDerivativeChainValue
    }

    override val maxResultShape: IntImmutableList
        get() = maxShape
}
