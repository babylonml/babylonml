package com.babylonml.backend.inference.operations

import com.babylonml.backend.common.TensorPointer
import com.babylonml.backend.inference.operations.tornadovm.TvmFloatArray
import com.babylonml.backend.tornadovm.TvmTensorOperations
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.TaskGraph
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

@Suppress("unused")
class RoPEOperation(name: String, qk: Operation, startPosition: Operation) : AbstractOperation(
    name,
    qk, startPosition
) {
    override fun doBuildTaskGraph(taskGraph: TaskGraph): TensorPointer {
        val qkPointer = leftPreviousOperation!!.buildTaskGraph(taskGraph)
        val startPositionPointer = rightPreviousOperation!!.buildTaskGraph(taskGraph)

        val cosPointer = buildCosTensor()
        val sinPointer = buildSinTensor()

        val resultPointer = executionContext.allocateResidentMemory(this, qkPointer.shape)
        val maxSequenceSize = maxResultShape.getInt(1)

        TvmTensorOperations.addRopeKernel(
            taskGraph, getTaskName(),
            qkPointer.floatBuffer(), qkPointer.shape, qkPointer.offset(),
            cosPointer.floatBuffer(), cosPointer.offset(),
            sinPointer.floatBuffer(), sinPointer.offset(),
            startPositionPointer.floatBuffer(), startPositionPointer.offset(),
            resultPointer.floatBuffer(), resultPointer.shape,
            resultPointer.offset(),
            maxSequenceSize
        )

        return resultPointer
    }

    override val maxResultShape: IntImmutableList
            by lazy {
                qk.maxResultShape
            }

    override val singlePassAllocations: List<IntImmutableList>
        get() = listOf(maxResultShape)
    override val localAllocations: List<IntImmutableList>
        get() = emptyList()
    override val inputAllocations: List<IntImmutableList>
        get() {
            val maxShape = maxResultShape
            val seqLen = maxShape.getInt(maxShape.size - 2)
            val dim = maxShape.getInt(maxShape.size - 1)

            return listOf(
                IntImmutableList.of(seqLen, dim),
                IntImmutableList.of(seqLen, dim)
            )
        }

    private fun buildCosTensor(): TensorPointer {
        val seqLen = maxResultShape.getInt(maxResultShape.size - 2)
        val dim = maxResultShape.getInt(maxResultShape.size - 1)
        val cosPointer =
            executionContext.allocateResidentMemory(this, IntImmutableList.of(dim))

        val buffer = cosPointer.buffer() as TvmFloatArray

        fillCosTensor(seqLen, dim, buffer)

        return cosPointer
    }

    internal fun fillCosTensor(
        seqLen: Int,
        dim: Int,
        buffer: TvmFloatArray
    ) {
        for (j in 0 until dim / 2) {
            val invFreq = 1.0 / 10_000.toDouble().pow(2 * j / dim)

            for (i in 1..seqLen) {
                val cosValue = (i * cos(invFreq)).toFloat()

                buffer[j] = cosValue
                buffer[j + dim / 2] = cosValue
            }
        }
    }

    private fun buildSinTensor(): TensorPointer {
        val seqLen = maxResultShape.getInt(maxResultShape.size - 2)
        val dimension = maxResultShape.getInt(maxResultShape.size - 1)
        val sinPointer =
            executionContext.allocateResidentMemory(this, IntImmutableList.of(dimension))

        val buffer = sinPointer.buffer() as TvmFloatArray
        return fillSinTensor(seqLen, dimension, buffer, sinPointer)
    }

    internal fun fillSinTensor(
        seqLen: Int,
        dim: Int,
        buffer: TvmFloatArray,
        sinPointer: TensorPointer
    ): TensorPointer {
        for (j in 0 until dim / 2) {
            val invFreq = 1.0 / 10_000.toDouble().pow(2 * j / dim)

            for (i in 1..seqLen) {
                val sinValue = (i * sin(invFreq)).toFloat()

                buffer[j] = sinValue
                buffer[j + dim / 2] = sinValue
            }
        }

        return sinPointer
    }
}