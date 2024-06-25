package com.babylonml.backend.training.optimizer

import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.cpu.VectorOperations
import com.babylonml.backend.training.execution.ContextInputSource
import com.babylonml.backend.training.execution.TrainingExecutionContext
import com.babylonml.backend.training.operations.MiniBatchListener
import com.babylonml.backend.training.operations.Operation
import it.unimi.dsi.fastutil.ints.IntImmutableList

class SimpleGradientDescentOptimizer(inputSource: ContextInputSource) : GradientOptimizer, MiniBatchListener {
    private var scaleValue = 1

    init {
        inputSource.addMiniBatchListener(this)
    }

    override fun optimize(
        executionContext: TrainingExecutionContext,
        matrix: FloatArray,
        matrixOffset: Int,
        shape: IntImmutableList,
        gradient: FloatArray,
        gradientOffset: Int,
        learningRate: Float,
        operation: Operation
    ) {
        val pointer = executionContext.allocateBackwardMemory(operation, shape)
        val buffer = pointer.buffer()
        val bufferOffset = pointer.offset()

        val stride = TensorOperations.stride(shape)
        VectorOperations.multiplyVectorToScalar(
            gradient, gradientOffset,
            -learningRate / scaleValue, buffer, bufferOffset,
            stride
        )
        VectorOperations.addVectorToVector(
            matrix, matrixOffset, buffer, bufferOffset, matrix, matrixOffset,
            stride
        )
    }

    override fun calculateRequiredMemoryAllocations(shape: IntImmutableList): List<IntImmutableList> {
        return listOf(shape)
    }

    override fun onMiniBatchStart(miniBatchIndex: Long, miniBatchSize: Int) {
        assert(miniBatchSize > 0)
        scaleValue = miniBatchSize
    }
}
