package com.babylonml.backend.inference.operations

import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.inference.operations.tornadovm.InferenceExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.TaskGraph

interface Operation {
    val maxResultShape: IntImmutableList

    fun execute(taskGraph: TaskGraph): TensorPointer

    val singlePassAllocations: List<IntImmutableList>

    val localAllocations: List<IntImmutableList>

    var leftPreviousOperation: Operation?

    var rightPreviousOperation: Operation?

    var nextOperation: Operation?

    fun prepareForNextPass()

    val executionContext: InferenceExecutionContext

    val name: String?
}
