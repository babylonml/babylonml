package com.babylonml.backend.operations

import com.babylonml.backend.ExecutionContext
import com.babylonml.backend.tensor.common.TensorPointer
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.GridScheduler
import uk.ac.manchester.tornado.api.TaskGraph

class I32InputSourceOperation(
    var value: IntArray,
    val shape: IntImmutableList,
    private val maxShape: IntImmutableList,
    name: String, executionContext: ExecutionContext,
) :
    AbstractOperation(name, executionContext, null, null) {

    constructor(
        value: IntArray,
        shape: IntImmutableList,
        name: String, executionContext: ExecutionContext
    ) : this(value, shape, shape, name, executionContext)

    private var dataPointer: TensorPointer? = null

    override val maxResultShape: IntImmutableList
        get() = maxShape

    override fun doBuildTaskGraph(taskGraph: TaskGraph, gridScheduler: GridScheduler): TensorPointer {
        val dataPointer = executionContext.allocateInputMemory(this, shape, TensorPointer.DType.INT32)

        val buffer = dataPointer.intBuffer()
        val offset = dataPointer.offset()

        for (i in value.indices) {
            buffer[i + offset] = value[i]
        }

        this.dataPointer = dataPointer
        return dataPointer
    }

    override val maxI32InputAllocations: List<IntImmutableList>
        get() = listOf(maxShape)

    override fun prepareForNextExecutionPass() {
        super.prepareForNextExecutionPass()

        dataPointer?.let {
            val buffer = it.intBuffer()
            val offset = it.offset()

            for (i in value.indices) {
                buffer[i + offset] = value[i]
            }
        }
    }
}