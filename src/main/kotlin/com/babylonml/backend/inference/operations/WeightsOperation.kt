package com.babylonml.backend.inference.operations

import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.inference.operations.tornadovm.InferenceExecutionContext
import com.babylonml.backend.inference.operations.tornadovm.TvmByteArray
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.TaskGraph


class WeightsOperation(
    override val name: String, executionContext: InferenceExecutionContext,
    private val data: ByteArray, val shape: IntImmutableList
) : AbstractOperation(name, executionContext, null, null) {
    override val residentAllocations: List<IntImmutableList>
        get() = listOf(shape)

    override val maxResultShape: IntImmutableList
        get() = IntImmutableList.of()

    override fun doBuildTaskGraph(taskGraph: TaskGraph): TensorPointer {
        val weightsPointer = executionContext.allocateResidentMemory(this, shape)

        val buffer = executionContext.getMemoryBuffer(weightsPointer.pointer) as TvmByteArray
        for (i in data.indices) {
            buffer[i] = data[i]
        }
        return weightsPointer
    }

    override val singlePassAllocations: List<IntImmutableList>
        get() = emptyList()

    override val localAllocations: List<IntImmutableList>
        get() = emptyList()

    override val inputAllocations: List<IntImmutableList>
        get() = emptyList()
}