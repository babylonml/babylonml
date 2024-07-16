package com.babylonml.backend.inference.operations

import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.tornadovm.TvmTensorOperations
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.TaskGraph
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

class RoPEOperation(name: String, qk: AbstractOperation, startPosition: AbstractOperation) : AbstractOperation(
    name,
    qk, startPosition
) {
    override fun doBuildTaskGraph(taskGraph: TaskGraph): TensorPointer {
        val qkPointer = leftPreviousOperation!!.buildTaskGraph(taskGraph)
        val startPositionPointer = rightPreviousOperation!!.buildTaskGraph(taskGraph)

        val headDimension = qkPointer.shape.getInt(3)
        val maxSeqLen = maxResultShape.getInt(1)

        val invFreqs = generateInvFreqs(headDimension)
        val cosPointer = buildCosTensor(maxSeqLen, headDimension, invFreqs)
        val sinPointer = buildSinTensor(maxSeqLen, headDimension, invFreqs)

        val resultPointer = executionContext.allocateSinglePassMemory(
            this, qkPointer.shape
        )

        TvmTensorOperations.addRopeKernel(
            taskGraph, getTaskName(),
            qkPointer.floatBuffer(), qkPointer.shape, qkPointer.offset(),
            cosPointer.floatBuffer(), cosPointer.offset(),
            sinPointer.floatBuffer(), sinPointer.offset(),
            startPositionPointer.intBuffer(), startPositionPointer.offset(),
            resultPointer.floatBuffer(), resultPointer.shape,
            resultPointer.offset(),
            maxSeqLen
        )

        return resultPointer
    }

    override val maxResultShape: IntImmutableList
            by lazy {
                //[bs, seqLen, numHeads, headDim]
                qk.maxResultShape
            }

    override val maxSinglePassAllocations: List<IntImmutableList>
        get() = listOf(maxResultShape)

    override val maxResidentF32Allocations: List<IntImmutableList>
        get() {
            val maxSequenceLen = maxResultShape.getInt(1)
            val headDimension = maxResultShape.getInt(3)

            return listOf(
                IntImmutableList.of(maxSequenceLen, headDimension / 2),
                IntImmutableList.of(maxSequenceLen, headDimension / 2)
            )
        }

    private fun buildCosTensor(maxSeqLen: Int, headDimension: Int, invFreqs: FloatArray): TensorPointer {
        val cosPointer =
            executionContext.allocateResidentMemory(
                this, IntImmutableList.of(maxSeqLen, headDimension / 2),
                TensorPointer.DType.F32
            )

        fillCosTensor(maxSeqLen, headDimension, invFreqs, cosPointer.floatBuffer(), cosPointer.offset())
        return cosPointer
    }

    internal fun generateInvFreqs(headDimension: Int): FloatArray {
        val halfDimension = headDimension / 2
        val invFreqs = FloatArray(halfDimension)

        for (j in 0 until halfDimension) {
            invFreqs[j] = (1.0 / 10_000.toDouble().pow(2.0 * j.toDouble() / headDimension.toDouble())).toFloat()
        }

        return invFreqs
    }

    internal fun fillCosTensor(
        maxSequenceLen: Int,
        headDimension: Int,
        invFreqs: FloatArray,
        buffer: TvmFloatArray,
        offset: Int
    ) {
        val halfDimension = headDimension / 2
        var position = 0

        for (i in 0 until maxSequenceLen) {
            for (j in 0 until halfDimension) {
                val invFreq = invFreqs[j]
                val cosValue = cos(i.toDouble() * invFreq).toFloat()

                buffer[position + offset] = cosValue
                position += 1
            }
        }
    }

    private fun buildSinTensor(
        maxSequenceLen: Int,
        headDimension: Int,
        invFreqs: FloatArray
    ): TensorPointer {

        val sinPointer = executionContext.allocateResidentMemory(
            this,
            IntImmutableList.of(maxSequenceLen, headDimension / 2), TensorPointer.DType.F32
        )

        fillSinTensor(maxSequenceLen, headDimension, invFreqs, sinPointer.floatBuffer(), sinPointer.offset())
        return sinPointer
    }

    internal fun fillSinTensor(
        maxSequenceLen: Int,
        headDimension: Int,
        invFreqs: FloatArray,
        buffer: TvmFloatArray,
        offset: Int
    ) {
        val halfDimension = headDimension / 2
        var position = 0

        for (i in 0 until maxSequenceLen) {
            for (j in 0 until halfDimension) {
                val invFreq = invFreqs[j]
                val sinValue = sin(i.toDouble() * invFreq).toFloat()

                buffer[offset + position] = sinValue
                position += 1
            }
        }
    }
}