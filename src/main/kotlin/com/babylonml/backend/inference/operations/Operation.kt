package com.babylonml.backend.inference.operations

import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.inference.operations.tornadovm.InferenceExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.TaskGraph

interface Operation {
    val maxResultShape: IntImmutableList

    fun buildTaskGraph(taskGraph: TaskGraph): TensorPointer

    val maxResidentInt8Allocations: List<IntImmutableList>
    val maxResidentF16Allocations : List<IntImmutableList>
    val maxResidentF32Allocations : List<IntImmutableList>

    val maxSinglePassAllocations: List<IntImmutableList>

    val maxLocalAllocations: List<IntImmutableList>

    val maxInputAllocations: List<IntImmutableList>

    var leftPreviousOperation: Operation?

    var rightPreviousOperation: Operation?

    var nextOperation: Operation?

    fun prepareForNextExecutionPass()

    val executionContext: InferenceExecutionContext

    val name: String?
}
