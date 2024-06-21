package com.babylonml.tensor

import org.apache.commons.rng.UniformRandomProvider


class FloatTensor internal constructor(val shape: IntArray, private val data: Array<Any>) {
    constructor(shape: IntArray) : this(shape, createData(shape))

    internal fun enumerateIndexes(): Sequence<IntArray> {
        return doEnumerateIndexes(shape)
    }


    operator fun times(value: Float): FloatTensor {
        val result = FloatTensor(shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) * value)
        }

        return result
    }

    operator fun times(other: Int): FloatTensor {
        val result = FloatTensor(shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) * other)
        }

        return result
    }

    operator fun div(other: Int): FloatTensor {
        val result = FloatTensor(shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) / other)
        }

        return result
    }

    operator fun div(other: Float): FloatTensor {
        val result = FloatTensor(shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) / other)
        }

        return result
    }

    operator fun plus(other: FloatTensor): FloatTensor {
        val (first, second) = broadcast(this, other)
        return first.doAdd(second)
    }

    operator fun minus(other: FloatTensor): FloatTensor {
        val (first, second) = broadcast(this, other)

        return first.doMinus(second)
    }

    private fun doMinus(second: FloatTensor): FloatTensor {
        val result = FloatTensor(shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) - second.get(indexes))
        }

        return result

    }

    @Suppress("unused")
    infix fun hdm(other: FloatTensor): FloatTensor {
        val (first, second) = broadcast(this, other)

        return first.doHadamard(second)
    }

    private fun doHadamard(second: FloatTensor): FloatTensor {
        val result = FloatTensor(shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) * second.get(indexes))
        }

        return result
    }

    private fun doAdd(other: FloatTensor): FloatTensor {
        val result = FloatTensor(shape)
        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) + other.get(indexes))
        }

        return result
    }

    internal fun get(indexes: IntArray): Float {
        return doGet(indexes, data) as Float
    }


    internal fun set(indexes: IntArray, value: Float) {
        doSet(indexes, data, value)
    }


    val size = calculateSize(shape)


    fun broadcast(vararg newShape: Int): FloatTensor {
        if (newShape.size < shape.size) {
            throw IllegalArgumentException("New shape must have at least as many dimensions as the original shape")
        }

        val modifiedCurrentShape = if (newShape.size == shape.size) {
            shape
        } else {
            val diff = newShape.size - shape.size
            val prefix = IntArray(diff) {
                1
            }
            prefix + shape
        }

        var newData = deepArrayCopy(data)
        for (i in 0 until modifiedCurrentShape.size - shape.size) {
            newData = Array(modifiedCurrentShape[i]) {
                newData
            }
        }


        for ((index, dimensions) in modifiedCurrentShape.zip(newShape).withIndex().reversed()) {
            val currentDimension = dimensions.first
            val newDimension = dimensions.second

            if (currentDimension != newDimension) {
                if (currentDimension == 1) {
                    if (index > 0) {
                        val activeDimensions = IntArray(index) {
                            modifiedCurrentShape[it]
                        }

                        for (indexes in doEnumerateIndexes(activeDimensions)) {
                            val value = doGet(indexes + intArrayOf(0), newData)
                            val newValue = broadcastObjectToArray(value, newDimension)

                            doSet(indexes, newData, newValue)
                        }
                    } else {
                        newData = broadcastObjectToArray(newData[0], newDimension)
                    }
                } else {
                    throw IllegalArgumentException(
                        "Cannot broadcast shape ${shape.joinToString(",", "[", "]")} " +
                                "to ${newShape.joinToString(",", "[", "]")}"
                    )
                }
            }
        }

        return FloatTensor(newShape, newData)
    }

    @Suppress("unused")
    fun reduce(vararg newShape: Int): FloatTensor {
        if (newShape.size > shape.size) {
            throw IllegalArgumentException("Original shape must have at least as many dimensions as the new shape")
        }

        val modifiedNewShape = if (newShape.size == shape.size) {
            newShape
        } else {
            val diff = shape.size - newShape.size
            val prefix = IntArray(diff) {
                1
            }
            prefix + newShape
        }

        for ((currentDimension, newDimension) in shape.zip(modifiedNewShape)) {
            if (newDimension != 1) {
                if (newDimension != currentDimension) {
                    throw IllegalArgumentException(
                        "Cannot reduce shape ${shape.joinToString(",", "[", "]")} " +
                                "to ${newShape.joinToString(",", "[", "]")}"
                    )
                }
            }
        }

        val tensor = FloatTensor(modifiedNewShape)

        for (index in enumerateIndexes()) {
            val reducedIndex = tensor.reducedIndex(index)
            val value = tensor.get(reducedIndex)

            tensor.set(reducedIndex, value + get(index))
        }

        if (newShape.size == shape.size) {
            return tensor
        }

        var newData = tensor.data
        for (i in 0 until shape.size - newShape.size) {
            if (newData.size != 1) {
                throw IllegalArgumentException(
                    "Cannot reduce" +
                            " shape ${shape.joinToString(",", "[", "]")} " +
                            " to ${newShape.joinToString(",", "[", "]")}"
                )
            }

            @Suppress("UNCHECKED_CAST")
            newData = newData[0] as Array<Any>
        }

        return FloatTensor(newShape, newData)
    }

    fun toFlatArray(): FloatArray {
        val result = FloatArray(size)

        for (indexes in enumerateIndexes()) {
            result[flattenIndex(indexes)] = get(indexes)
        }

        return result
    }

    private fun reducedIndex(indexes: IntArray): IntArray {
        return indexes.mapIndexed() { index, value -> if (value >= shape[index]) 0 else value }.toIntArray()
    }

    private fun flattenIndex(indexes: IntArray): Int {
        var result = 0
        var multiplier = 1

        for (i in indexes.size - 1 downTo 0) {
            result += indexes[i] * multiplier
            multiplier *= shape[i]
        }

        return result

    }

    companion object {
        private fun doSet(indexes: IntArray, currentData: Array<Any>, value: Any) {
            var data = currentData
            for (i in 0 until indexes.size - 1) {
                @Suppress("UNCHECKED_CAST")
                data = data[indexes[i]] as Array<Any>
            }

            data[indexes[indexes.size - 1]] = value
        }

        private fun deepArrayCopy(data: Array<Any>): Array<Any> {
            return if (data[0] is Array<*>) {
                Array(data.size) {
                    @Suppress("UNCHECKED_CAST")
                    deepArrayCopy(data[it] as Array<Any>)
                }
            } else {
                data.copyOf()
            }
        }

        private fun doGet(indexes: IntArray, currentData: Array<Any>): Any {
            var data = currentData

            for (i in 0 until indexes.size - 1) {
                @Suppress("UNCHECKED_CAST")
                data = data[indexes[i]] as Array<Any>
            }

            return data[indexes.last()]
        }

        private fun doEnumerateIndexes(shape: IntArray): Sequence<IntArray> {
            return sequence {
                val indexes = IntArray(shape.size)
                var i = 0

                while (i < calculateSize(shape)) {
                    yield(indexes.copyOf())

                    i++
                    for (j in shape.size - 1 downTo 0) {
                        indexes[j]++
                        if (indexes[j] < shape[j]) {
                            break
                        } else {
                            indexes[j] = 0
                        }
                    }
                }
            }
        }

        private fun calculateSize(shape: IntArray) = shape.reduce { acc, i -> acc * i }

        fun random(source: UniformRandomProvider, vararg shape: Int): FloatTensor {
            val result = FloatTensor(shape)

            for (indexes in result.enumerateIndexes()) {
                result.set(indexes, source.nextFloat(-1.0f, 1.0f))
            }

            return result
        }

        @Suppress("unused")
        fun natural(vararg  shape: Int): FloatTensor {
            val result = FloatTensor(shape)
            var i = 0

            for (indexes in result.enumerateIndexes()) {
                result.set(indexes, i.toFloat())
                i++
            }

            return result
        }

        @Suppress("unused")
        fun constant(value: Float, vararg shape: Int): FloatTensor {
            val result = FloatTensor(shape)

            for (indexes in result.enumerateIndexes()) {
                result.set(indexes, value)
            }

            return result
        }

        private fun broadcastObjectToArray(value: Any, size: Int): Array<Any> {
            val result = Array(size) {
                value
            }

            return result
        }

        private fun createData(shape: IntArray): Array<Any> {
            return if (shape.size == 1) {
                Array(shape[0]) {
                    0.0f
                }
            } else {
                Array(shape[0]) {
                    createData(shape.sliceArray(1 until shape.size))
                }
            }
        }

        private fun broadcast(first: FloatTensor, second: FloatTensor): Pair<FloatTensor, FloatTensor> {
            val firstShape = first.shape
            val secondShape = second.shape

            val maxShape = if (firstShape.size > secondShape.size) {
                firstShape
            } else {
                secondShape
            }

            val firstBroadcast = first.broadcast(*maxShape)
            val secondBroadcast = second.broadcast(*maxShape)

            return Pair(firstBroadcast, secondBroadcast)
        }
    }
}

operator fun Int.times(other: FloatTensor): FloatTensor {
    return other * this
}

operator fun Float.times(other: FloatTensor): FloatTensor {
    return other * this
}

operator fun Float.div(other: FloatTensor): FloatTensor {
    val result = FloatTensor(other.shape)

    for (indexes in other.enumerateIndexes()) {
        result.set(indexes, this / result.get(indexes))
    }

    return result
}