package com.babylonml.backend.operations

import com.babylonml.backend.ExecutionContext
import com.babylonml.backend.TvmArray
import com.babylonml.backend.TvmFloatArray
import com.babylonml.backend.TvmIntArray
import com.babylonml.backend.tensor.common.TensorPointer
import com.babylonml.backend.tensor.tornadovm.TvmCommons
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.GridScheduler
import uk.ac.manchester.tornado.api.TaskGraph

abstract class AbstractOperation(
    val name: String,
    executionContext: ExecutionContext?,
    var leftPreviousOperation: AbstractOperation?,
    var rightPreviousOperation: AbstractOperation?
) {
    var executionContext: ExecutionContext

    open val maxSinglePassAllocations: List<IntImmutableList>
        get() = emptyList()
    open val maxLocalAllocations: List<IntImmutableList>
        get() = emptyList()

    open val maxF32InputAllocations: List<IntImmutableList>
        get() = emptyList()
    open val maxI32InputAllocations: List<IntImmutableList>
        get() = emptyList()

    open val maxResidentInt8Allocations: List<IntImmutableList>
        get() = emptyList()
    open val maxResidentF16Allocations: List<IntImmutableList>
        get() = emptyList()
    open val maxResidentF32Allocations: List<IntImmutableList>
        get() = emptyList()

    abstract val maxResultShape: IntImmutableList

    var nextOperation: AbstractOperation? = null

    constructor(
        name: String,
        leftOperation: AbstractOperation?,
        rightOperation: AbstractOperation?
    ) : this(name, null, leftOperation, rightOperation)

    init {
        leftPreviousOperation?.let {
            require(it.nextOperation == null) { "Left operation already has a next operation" }
            it.nextOperation = this
        }

        rightPreviousOperation?.let {
            require(it.nextOperation == null) { "Right operation already has a next operation" }
            it.nextOperation = this
        }

        if (executionContext == null) {
            if (leftPreviousOperation != null) {
                this.executionContext = leftPreviousOperation!!.executionContext
            } else if (rightPreviousOperation != null) {
                this.executionContext = rightPreviousOperation!!.executionContext
            } else {
                throw IllegalArgumentException("At least one of the operations should be provided")
            }
        } else {
            this.executionContext = executionContext
        }
    }

    fun buildTaskGraph(taskGraph: TaskGraph, gridScheduler: GridScheduler): TensorPointer {
        prepareForNextExecutionPass()
        return doBuildTaskGraph(taskGraph, gridScheduler)
    }

    abstract fun doBuildTaskGraph(taskGraph: TaskGraph, gridScheduler: GridScheduler): TensorPointer

    open fun prepareForNextExecutionPass() {
        leftPreviousOperation?.prepareForNextExecutionPass()
        rightPreviousOperation?.prepareForNextExecutionPass()
    }

    fun TensorPointer.buffer(): TvmArray {
        return executionContext.getMemoryBuffer(this)
    }

    fun TensorPointer.floatBuffer(): TvmFloatArray {
        if (dtype != TensorPointer.DType.F32) {
            throw IllegalArgumentException("Tensor is not of type F32")
        }

        return buffer() as TvmFloatArray
    }

    fun TensorPointer.intBuffer(): TvmIntArray {
        if (dtype != TensorPointer.DType.INT32) {
            throw IllegalArgumentException("Tensor is not of type I32")
        }

        return buffer() as TvmIntArray
    }

    fun TensorPointer.offset() = pointer.toInt()

    fun getTaskName(prefix: String? = null): String {
        return TvmCommons.generateName(
            prefix ?: ("" + name)
        )
    }
}
