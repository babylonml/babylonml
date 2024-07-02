package com.babylonml.backend.inference.operations

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.tornadovm.TvmVectorOperations
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.TaskGraph

class Add(name: String?, leftOperation: Operation, rightOperation: Operation) :
    AbstractOperation(name, leftOperation, rightOperation) {

    constructor(leftOperation: Operation, rightOperation: Operation) : this(null, leftOperation, rightOperation)

    override fun execute(taskGraph: TaskGraph): TensorPointer {
        val leftOperandPointer = leftPreviousOperation!!.execute(taskGraph)
        val rightOperandPointer = rightPreviousOperation!!.execute(taskGraph)

        return broadcastIfNeeded(
            taskGraph, leftOperandPointer, rightOperandPointer
        ) { leftPointer, rightPointer, resultPointer ->
            TvmVectorOperations.addVectorToVector(
                leftPointer.buffer(), leftPointer.offset(), rightPointer.buffer(),
                rightPointer.offset(), resultPointer.buffer(),
                resultPointer.offset(), CommonTensorOperations.stride(resultPointer.shape)
            )
        }
    }

    override val localAllocations: List<IntImmutableList>
        get() = emptyList()

    override val singlePassAllocations: List<IntImmutableList>
        get() = listOf(maxResultShape)

    override val maxResultShape: IntImmutableList by lazy {
        CommonTensorOperations.calculateMaxShape(
            leftPreviousOperation!!.maxResultShape,
            rightPreviousOperation!!.maxResultShape
        )
    }
}
