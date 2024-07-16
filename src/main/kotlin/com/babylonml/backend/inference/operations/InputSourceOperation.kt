package com.babylonml.backend.inference.operations

import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.inference.operations.tornadovm.InferenceExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.TaskGraph

class InputSourceOperation(
    var value: FloatArray,
    val shape: IntImmutableList,
    name: String, executionContext: InferenceExecutionContext,
) :
    AbstractOperation(name, executionContext, null, null) {

    private var dataPointer: TensorPointer? = null

    override val maxResultShape: IntImmutableList
        get() = shape

    override fun doBuildTaskGraph(taskGraph: TaskGraph): TensorPointer {
        val dataPointer = executionContext.allocateInputMemory(this, shape)

        val buffer = dataPointer.floatBuffer()
        val offset = dataPointer.offset()

        for (i in value.indices) {
            buffer[i + offset] = value[i]
        }
        this.dataPointer = dataPointer

        return dataPointer
    }

    override val inputAllocations: List<IntImmutableList>
        get() = listOf(shape)

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