package com.babylonml.backend.inference.operations

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.inference.operations.tornadovm.TvmFloatArray
import com.babylonml.backend.tornadovm.TvmVectorOperations
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.TaskGraph

class AddOperation(name: String?, leftOperation: Operation, rightOperation: Operation) :
    AbstractOperation(name, leftOperation, rightOperation) {

    override fun doBuildTaskGraph(taskGraph: TaskGraph): TensorPointer {
        val leftOperandPointer = leftPreviousOperation!!.buildTaskGraph(taskGraph)
        val rightOperandPointer = rightPreviousOperation!!.buildTaskGraph(taskGraph)

        return broadcastIfNeeded(
            taskGraph, leftOperandPointer, rightOperandPointer
        ) { leftPointer, rightPointer, resultPointer ->
            TvmVectorOperations.addVectorToVectorTask(
                taskGraph,
                getTaskName(),
                leftPointer.buffer(),
                leftPointer.offset(),
                rightPointer.buffer(),
                rightPointer.offset(),
                resultPointer.buffer() as TvmFloatArray,
                resultPointer.offset(),
                CommonTensorOperations.stride(resultPointer.shape)
            )
        }
    }

    override val inputAllocations: List<IntImmutableList>
        get() = emptyList()

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
