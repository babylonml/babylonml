package com.babylonml.backend.inference.operations

import com.babylonml.backend.common.TensorPointer
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.TaskGraph

class F32InputSourceOperation(
    var value: FloatArray,
    val shape: IntImmutableList,
    private val maxShape: IntImmutableList,
    name: String, executionContext: InferenceExecutionContext,
) :
    AbstractOperation(name, executionContext, null, null) {

    constructor(
        value: FloatArray,
        shape: IntImmutableList,
        name: String, executionContext: InferenceExecutionContext
    ) : this(value, shape, shape, name, executionContext)

    private var dataPointer: TensorPointer? = null

    override val maxResultShape: IntImmutableList
        get() = maxShape

    override fun doBuildTaskGraph(taskGraph: TaskGraph): TensorPointer {
        val dataPointer = executionContext.allocateInputMemory(this, shape, TensorPointer.DType.F32)

        val buffer = dataPointer.floatBuffer()
        val offset = dataPointer.offset()

        for (i in value.indices) {
            buffer[i + offset] = value[i]
        }

        this.dataPointer = dataPointer
        return dataPointer
    }

    override val maxF32InputAllocations: List<IntImmutableList>
        get() = listOf(maxShape)

    override fun prepareForNextExecutionPass() {
        super.prepareForNextExecutionPass()

        dataPointer?.let {
            val buffer = it.floatBuffer()
            val offset = it.offset()

            for (i in value.indices) {
                buffer[i + offset] = value[i]
            }
        }
    }
}