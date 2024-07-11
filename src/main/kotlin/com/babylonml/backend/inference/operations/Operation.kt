package com.babylonml.backend.inference.operations

import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.inference.operations.tornadovm.InferenceExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.TaskGraph

interface Operation {
    val maxResultShape: IntImmutableList

    fun buildTaskGraph(taskGraph: TaskGraph): TensorPointer

    val residentAllocations: List<IntImmutableList>

    val singlePassAllocations: List<IntImmutableList>

    val localAllocations: List<IntImmutableList>

    val inputAllocations: List<IntImmutableList>

    var leftPreviousOperation: Operation?

    var rightPreviousOperation: Operation?

    var nextOperation: Operation?

    fun prepareForNextExecutionPass()

    val executionContext: InferenceExecutionContext

    val name: String?
}
