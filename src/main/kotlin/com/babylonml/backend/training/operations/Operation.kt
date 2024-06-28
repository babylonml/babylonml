package com.babylonml.backend.training.operations

import com.babylonml.backend.training.execution.TensorPointer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import it.unimi.dsi.fastutil.ints.IntImmutableList

interface Operation {
    val maxResultShape: IntImmutableList

    fun forwardPassCalculation(): TensorPointer

    fun leftBackwardDerivativeChainValue(): TensorPointer

    fun rightBackwardDerivativeChainValue(): TensorPointer

    val forwardMemoryAllocations: List<IntImmutableList>

    val backwardMemoryAllocations: List<IntImmutableList>

    var leftPreviousOperation: Operation?

    var rightPreviousOperation: Operation?

    var nextOperation: Operation?

    fun clearNextOperation()

    var derivativeChainPointer: TensorPointer?

    val requiresBackwardDerivativeChainValue: Boolean

    fun prepareForNextPropagation()

    fun startEpochExecution()

    val executionContext: TrainingExecutionContext

    val name: String?
}
