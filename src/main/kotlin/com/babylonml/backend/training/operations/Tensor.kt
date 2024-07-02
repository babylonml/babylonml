package com.babylonml.backend.training.operations

import com.babylonml.backend.cpu.TensorOperations
import it.unimi.dsi.fastutil.ints.IntImmutableList

class Tensor(data: FloatArray, shape: IntArray) {
    @JvmField
    val data: FloatArray
    @JvmField
    val shape: IntImmutableList

    init {
        val stride = TensorOperations.stride(shape)

        require(data.size == stride) { "Data length must be equal to the stride of the shape." }

        this.data = data
        this.shape = IntImmutableList.of(*shape)
    }

    constructor(shape: IntArray) : this(FloatArray(TensorOperations.stride(shape)), shape)

    val isEmpty: Boolean
        get() = data.isEmpty()

    fun size(): Int {
        return data.size
    }


    companion object {
        @Suppress("unused")
        fun fromVector(data: FloatArray): Tensor {
            val shape = intArrayOf(data.size)
            return Tensor(data, shape)
        }

        @JvmStatic
        fun fromMatrix(data: Array<FloatArray>): Tensor {
            val shape = intArrayOf(data.size, data[0].size)
            val tensorData = FloatArray(TensorOperations.stride(shape))

            for (i in data.indices) {
                System.arraycopy(data[i], 0, tensorData, i * data[i].size, data[i].size)
            }

            return Tensor(tensorData, shape)
        }
    }
}
