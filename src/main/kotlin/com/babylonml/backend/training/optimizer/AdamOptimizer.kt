package com.babylonml.backend.training.optimizer

import com.babylonml.backend.common.CommonTensorOperations
import com.babylonml.backend.cpu.VectorOperations
import com.babylonml.backend.training.execution.ContextInputSource
import com.babylonml.backend.training.execution.TrainingExecutionContext
import com.babylonml.backend.training.operations.MiniBatchListener
import com.babylonml.backend.training.operations.Operation
import it.unimi.dsi.fastutil.ints.IntImmutableList
import kotlin.math.pow

class AdamOptimizer(
    private val beta1: Float,
    private val beta2: Float,
    private val epsilon: Float,
    inputSource: ContextInputSource
) : GradientOptimizer, MiniBatchListener {
    private lateinit var avgMovement: FloatArray
    private lateinit var avgMovementSqr: FloatArray


    private var batchIndex: Long = 0
    private var scaleValue = 1

    constructor(inputSource: ContextInputSource) : this(DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON, inputSource)

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
        val stride = CommonTensorOperations.stride(shape)

        val avgMovementPointer = executionContext.allocateBackwardMemory(operation, shape)
        val avgMovementBuffer = executionContext.getMemoryBuffer(avgMovementPointer.pointer)
        val avgMovementBufferOffset = TrainingExecutionContext.addressOffset(avgMovementPointer.pointer)


        val avgMovementSqrPointer = executionContext.allocateBackwardMemory(operation, shape)
        val avgMovementSqrBuffer = executionContext.getMemoryBuffer(avgMovementSqrPointer.pointer)
        val avgMovementSqrBufferOffset = TrainingExecutionContext.addressOffset(avgMovementSqrPointer.pointer)

        updateAvgMovement(
            gradient, gradientOffset, avgMovement, avgMovementSqr,
            avgMovementBuffer, avgMovementBufferOffset, stride, beta1, beta2, scaleValue
        )
        movingAverageBiasCorrection(
            avgMovement, beta1, batchIndex, avgMovementBuffer, avgMovementBufferOffset,
            stride
        )
        movingAverageBiasCorrection(
            avgMovementSqr, beta2, batchIndex, avgMovementSqrBuffer, avgMovementSqrBufferOffset,
            stride
        )

        calculateCorrections(
            avgMovementBuffer, avgMovementBufferOffset, avgMovementSqrBuffer,
            avgMovementSqrBufferOffset, stride, learningRate, epsilon
        )
        VectorOperations.addVectorToVector(
            matrix, matrixOffset, avgMovementBuffer, avgMovementBufferOffset, matrix,
            matrixOffset, stride
        )
    }

    override fun calculateRequiredMemoryAllocations(shape: IntImmutableList): List<IntImmutableList> {
        val stride = CommonTensorOperations.stride(shape)

        this.avgMovement = FloatArray(stride)
        this.avgMovementSqr = FloatArray(stride)

        return listOf(shape, shape)
    }

    override fun onMiniBatchStart(miniBatchIndex: Long, miniBatchSize: Int) {
        assert(miniBatchSize > 0)

        batchIndex = miniBatchIndex
        scaleValue = miniBatchSize
    }

    companion object {
        const val DEFAULT_BETA1: Float = 0.9f
        const val DEFAULT_BETA2: Float = 0.999f
        const val DEFAULT_EPSILON: Float = 1e-8f

        fun updateAvgMovement(
            gradient: FloatArray?, gradientOffset: Int, avgMovement: FloatArray?,
            avgMovementSqr: FloatArray?, avgMovementBuffer: FloatArray?,
            avgMovementBufferOffset: Int, size: Int, beta1: Float, beta2: Float, scale: Int
        ) {
            //g = g / scale
            //w[n] = b1 * w[n-1] + (1-b1) * g
            VectorOperations.multiplyVectorToScalar(
                avgMovement, 0, beta1,
                avgMovement, 0,
                size
            )
            VectorOperations.multiplyVectorToScalar(
                gradient, gradientOffset,
                (1 - beta1) / scale, avgMovementBuffer, avgMovementBufferOffset,
                size
            )
            VectorOperations.addVectorToVector(
                avgMovement, 0, avgMovementBuffer, avgMovementBufferOffset,
                avgMovement, 0, size
            )
            //v[n] = b2 * v[n-1] + (1-b2) * g^2
            VectorOperations.multiplyVectorToScalar(
                avgMovementSqr,
                0, beta2, avgMovementSqr, 0, size
            )
            VectorOperations.vectorToVectorElementWiseMultiplication(
                gradient, gradientOffset, gradient,
                gradientOffset, avgMovementBuffer, avgMovementBufferOffset, size
            )
            VectorOperations.multiplyVectorToScalar(
                avgMovementBuffer, avgMovementBufferOffset,
                (1 - beta2) / (scale * scale),
                avgMovementBuffer, avgMovementBufferOffset, size
            )
            VectorOperations.addVectorToVector(
                avgMovementSqr, 0, avgMovementBuffer, avgMovementBufferOffset,
                avgMovementSqr, 0, size
            )
        }

        private fun movingAverageBiasCorrection(
            movingAverage: FloatArray?,
            betta: Float, iteration: Long, result: FloatArray, resultOffset: Int, size: Int
        ) {
            val coefficient: Float = (1.0 / (1.0 - betta.toDouble().pow(iteration.toDouble()))).toFloat()
            VectorOperations.multiplyVectorToScalar(
                movingAverage, 0, coefficient, result, resultOffset,
                size
            )
        }

        fun calculateCorrections(
            correctedMovingAverage: FloatArray?, correctedMovingAverageOffset: Int,
            correctedMovingAverageSqr: FloatArray?, correctedMovingAverageSqrOffset: Int,
            size: Int,
            learningRate: Float, epsilon: Float
        ) {
            VectorOperations.vectorElementsSqrt(
                correctedMovingAverageSqr, correctedMovingAverageSqrOffset,
                correctedMovingAverageSqr, correctedMovingAverageSqrOffset,
                size
            )
            VectorOperations.addScalarToVector(
                epsilon, correctedMovingAverageSqr, correctedMovingAverageSqrOffset,
                correctedMovingAverageSqr, correctedMovingAverageSqrOffset,
                size
            )
            VectorOperations.divideScalarOnVectorElements(
                -learningRate, correctedMovingAverageSqr, correctedMovingAverageSqrOffset,
                correctedMovingAverageSqr, correctedMovingAverageSqrOffset, size
            )
            VectorOperations.vectorToVectorElementWiseMultiplication(
                correctedMovingAverage, correctedMovingAverageOffset,
                correctedMovingAverageSqr, correctedMovingAverageSqrOffset, correctedMovingAverage,
                correctedMovingAverageOffset,
                size
            )
        }
    }
}
