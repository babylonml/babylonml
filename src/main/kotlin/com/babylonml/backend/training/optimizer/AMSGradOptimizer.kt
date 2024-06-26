package com.babylonml.backend.training.optimizer

import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.cpu.VectorOperations
import com.babylonml.backend.training.execution.ContextInputSource
import com.babylonml.backend.training.execution.TrainingExecutionContext
import com.babylonml.backend.training.operations.MiniBatchListener
import com.babylonml.backend.training.operations.Operation
import it.unimi.dsi.fastutil.ints.IntImmutableList

class AMSGradOptimizer(
    private val beta1: Float,
    private val beta2: Float,
    private val epsilon: Float,
    inputSource: ContextInputSource
) : GradientOptimizer, MiniBatchListener {
    private lateinit var avgMovement: FloatArray
    private lateinit var avgMovementSqr: FloatArray
    private lateinit var correctedAvgMovementSqr: FloatArray

    private var scaleValue = 1

    constructor(inputSource: ContextInputSource) : this(DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON, inputSource)

    init {
        inputSource.addMiniBatchListener(this)
    }

    override fun onMiniBatchStart(miniBatchIndex: Long, miniBatchSize: Int) {
        scaleValue = miniBatchSize
    }

    override fun optimize(
        executionContext: TrainingExecutionContext, matrix: FloatArray, matrixOffset: Int,
        shape: IntImmutableList, gradient: FloatArray, gradientOffset: Int, learningRate: Float, operation: Operation
    ) {
        val stride = TensorOperations.stride(shape)

        val calculationBufferPointer = executionContext.allocateBackwardMemory(operation, shape)
        val calculationBuffer = executionContext.getMemoryBuffer(calculationBufferPointer.pointer)
        val calculationBufferOffset = TrainingExecutionContext.addressOffset(calculationBufferPointer.pointer)

        AdamOptimizer.updateAvgMovement(
            gradient, gradientOffset, avgMovement, avgMovementSqr, calculationBuffer,
            calculationBufferOffset, stride, beta1, beta2, scaleValue
        )
        correctAvgMovementSqr(
            avgMovementSqr, correctedAvgMovementSqr, stride
        )
        calculateCorrections(
            avgMovement, correctedAvgMovementSqr, calculationBuffer, calculationBufferOffset,
            stride, learningRate, epsilon
        )
        VectorOperations.addVectorToVector(
            matrix, 0, calculationBuffer, calculationBufferOffset, matrix, 0,
            stride
        )
    }

    override fun calculateRequiredMemoryAllocations(shape: IntImmutableList): List<IntImmutableList> {
        val stride = TensorOperations.stride(shape)
        avgMovement = FloatArray(stride)
        avgMovementSqr = FloatArray(stride)
        correctedAvgMovementSqr = FloatArray(stride)

        return listOf(shape)
    }

    companion object {
        const val DEFAULT_BETA1: Float = 0.9f
        const val DEFAULT_BETA2: Float = 0.999f
        const val DEFAULT_EPSILON: Float = 1e-8f

        private fun correctAvgMovementSqr(
            avgMovementSqr: FloatArray, correctedAvgMovementSqr: FloatArray?,
            size: Int
        ) {
            VectorOperations.maxBetweenVectorElements(
                avgMovementSqr, 0, correctedAvgMovementSqr,
                0,
                correctedAvgMovementSqr, 0, size
            )
        }

        private fun calculateCorrections(
            movingAverage: FloatArray, correctedMovingAverageSqr: FloatArray,
            calculationBuffer: FloatArray, calculationBufferOffset: Int,
            size: Int, learningRate: Float, epsilon: Float
        ) {
            VectorOperations.vectorElementsSqrt(
                correctedMovingAverageSqr, 0,
                calculationBuffer, calculationBufferOffset, size
            )
            VectorOperations.addScalarToVector(
                epsilon, calculationBuffer, calculationBufferOffset, calculationBuffer,
                calculationBufferOffset, size
            )
            VectorOperations.divideScalarOnVectorElements(
                -learningRate, calculationBuffer, calculationBufferOffset,
                calculationBuffer, calculationBufferOffset, size
            )
            VectorOperations.vectorToVectorElementWiseMultiplication(
                movingAverage, 0,
                calculationBuffer, calculationBufferOffset,
                calculationBuffer, calculationBufferOffset, size
            )
        }
    }
}
