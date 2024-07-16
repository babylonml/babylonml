package com.babylonml.tensor

import com.babylonml.backend.inference.operations.TvmFloatArray
import org.apache.commons.rng.UniformRandomProvider
import kotlin.math.pow

class FloatTensor {
    private val data: Array<Any>

    val shape: IntArray
    val size: Int

    constructor(vararg shape: Int) : this(shape, createData(shape))

    internal constructor(shape: IntArray, data: Array<Any>) {
        for (dimension in shape) {
            if (dimension <= 0) {
                throw IllegalArgumentException("All dimensions must be positive")
            }
        }

        this.shape = shape
        this.data = data
        this.size = calculateSize(shape)
    }

    internal fun enumerateIndexes(): Sequence<IntArray> {
        return doEnumerateIndexes(shape)
    }

    operator fun times(value: Float): FloatTensor {
        val result = FloatTensor(*shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) * value)
        }

        return result
    }

    fun combineWith(
        other: FloatTensor,
        producer: (firstItem: Float, secondItem: Float) -> Float
    ): FloatTensor {
        val result = FloatTensor(*(shape + other.shape))

        val resultIndexes = result.enumerateIndexes().iterator()
        for (indexes in enumerateIndexes()) {
            for (otherIndexes in other.enumerateIndexes()) {
                result.set(resultIndexes.next(), producer(get(indexes), other.get(otherIndexes)))
            }

        }

        return result
    }

    operator fun times(other: Int): FloatTensor {
        val result = FloatTensor(*shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) * other)
        }

        return result
    }

    operator fun div(other: Int): FloatTensor {
        val result = FloatTensor(*shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) / other)
        }

        return result
    }

    operator fun div(other: Float): FloatTensor {
        val result = FloatTensor(*shape)

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
        val result = FloatTensor(*shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) - second.get(indexes))
        }

        return result

    }

    operator fun times(other: FloatTensor): FloatTensor {
        val (first, second) = broadcast(this, other)

        return first.doTimes(second)
    }

    private fun doTimes(second: FloatTensor): FloatTensor {
        val result = FloatTensor(*shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) * second.get(indexes))
        }

        return result
    }

    private fun doAdd(other: FloatTensor): FloatTensor {
        val result = FloatTensor(*shape)
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

    fun broadcast(vararg newShape: Int, till: Int = shape.size): FloatTensor {
        if (newShape.size < shape.size) {
            throw IllegalArgumentException("New shape must have at least as many dimensions as the original shape")
        }

        var broadcastTillRank = till
        val modifiedCurrentShape = if (newShape.size == shape.size) {
            shape
        } else {
            val diff = newShape.size - shape.size
            val prefix = IntArray(diff) {
                1
            }
            prefix + shape
        }
        broadcastTillRank += modifiedCurrentShape.size - shape.size

        var modifiedData = deepArrayCopy(data)
        for (i in 0 until modifiedCurrentShape.size - shape.size) {
            modifiedData = Array(modifiedCurrentShape[i]) {
                modifiedData
            }
        }

        val modifiedNewShape = if (newShape.size == broadcastTillRank) {
            newShape
        } else {
            IntArray(newShape.size) {
                if (it < broadcastTillRank) {
                    newShape[it]
                } else {
                    modifiedCurrentShape[it]
                }
            }
        }

        for ((index, dimensions) in modifiedCurrentShape.zip(modifiedNewShape).withIndex()
            .reversed()) {
            val currentDimension = dimensions.first
            val newDimension = dimensions.second

            if (currentDimension != newDimension) {
                if (currentDimension == 1) {
                    if (index > 0) {
                        val activeDimensions = IntArray(index) {
                            modifiedCurrentShape[it]
                        }

                        for (indexes in doEnumerateIndexes(activeDimensions)) {
                            val value = doGet(indexes + intArrayOf(0), modifiedData)
                            val newValue = broadcastObjectToArray(value, newDimension)

                            doSet(indexes, modifiedData, newValue)
                        }
                    } else {
                        modifiedData = broadcastObjectToArray(modifiedData[0], newDimension)
                    }
                } else {
                    throw IllegalArgumentException(
                        "Cannot broadcast shape ${shape.joinToString(",", "[", "]")} " +
                                "to ${newShape.joinToString(",", "[", "]")}"
                    )
                }
            }
        }

        return FloatTensor(modifiedNewShape, modifiedData)
    }

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

        val tensor = FloatTensor(*modifiedNewShape)

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

    fun unsquize(index: Int): FloatTensor {
        val newShape = shape.toMutableList()
        newShape.add(index, 1)

        val result = FloatTensor(*newShape.toIntArray())
        for (indexes in enumerateIndexes()) {
            val newIndex = indexes.toMutableList()
            newIndex.add(index, 0)
            result.set(newIndex.toIntArray(), get(indexes))
        }

        return result
    }

    fun slice(vararg slices: IntRange): FloatTensor {
        val broadcastSlices = if (slices.size < shape.size) {
            val diff = shape.size - slices.size
            val prefix = Array(diff) {
                0 until shape[it]
            }
            prefix + slices
        } else {
            slices
        }

        val newShape = IntArray(broadcastSlices.size) {
            broadcastSlices[it].endInclusive - broadcastSlices[it].start + 1
        }

        val result = FloatTensor(*newShape)

        for (indexes in result.enumerateIndexes()) {
            val originalIndexes = IntArray(shape.size) {
                broadcastSlices[it].start + indexes[it]
            }

            result.set(indexes, get(originalIndexes))
        }

        return result
    }

    fun cat(tensor: FloatTensor, dim: Int = shape.size - 1): FloatTensor {
        val newShape = shape.copyOf()

        newShape[dim] += tensor.shape[dim]
        val result = FloatTensor(*newShape)

        for (indexes in result.enumerateIndexes()) {
            val value = if (indexes[dim] < shape[dim]) {
                get(indexes)
            } else {
                tensor.get(indexes.mapIndexed { index, value ->
                    if (index == dim) {
                        value - shape[index]
                    } else {
                        value
                    }
                }.toIntArray())
            }

            result.set(indexes, value)
        }

        return result
    }

    fun sin(): FloatTensor {
        val result = FloatTensor(*shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, kotlin.math.sin(get(indexes)))
        }

        return result
    }

    fun cos(): FloatTensor {
        val result = FloatTensor(*shape)

        for (indexes in enumerateIndexes()) {
            result.set(indexes, kotlin.math.cos(get(indexes)))
        }

        return result
    }

    fun toFlatArray(): FloatArray {
        val result = FloatArray(size)

        for (indexes in enumerateIndexes()) {
            result[flattenIndex(indexes)] = get(indexes)
        }

        return result
    }

    fun toTvmFlatArray(length: Int = size, offset: Int = 0): TvmFloatArray {
        val flatArray = toFlatArray()
        val result = FloatArray(length + offset)
        System.arraycopy(flatArray, 0, result, offset, length)

        return TvmFloatArray.fromArray(result)
    }

    private fun reducedIndex(indexes: IntArray): IntArray {
        return indexes.mapIndexed() { index, value -> if (value >= shape[index]) 0 else value }
            .toIntArray()
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

    override fun toString(): String {
        return "FloatTensor(${shape.joinToString(",", "[", "])")}"
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
            val result = FloatTensor(*shape)

            for (indexes in result.enumerateIndexes()) {
                result.set(indexes, source.nextFloat(-1.0f, 1.0f))
            }

            return result
        }

        fun natural(vararg shape: Int): FloatTensor {
            val result = FloatTensor(*shape)
            var i = 1

            for (indexes in result.enumerateIndexes()) {
                result.set(indexes, i.toFloat())
                i++
            }

            return result
        }

        @Suppress("unused")
        fun constant(value: Float, vararg shape: Int): FloatTensor {
            val result = FloatTensor(*shape)

            for (indexes in result.enumerateIndexes()) {
                result.set(indexes, value)
            }

            return result
        }

        fun arrange(start: Int = 0, end: Int, step: Int = 1): FloatTensor {
            val result = FloatTensor((end - start) / step)

            for (i in start until end step step) {
                result.set(intArrayOf((i - start) / step), i.toFloat())
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

        private fun broadcast(
            first: FloatTensor,
            second: FloatTensor
        ): Pair<FloatTensor, FloatTensor> {
            val firstShape = first.shape
            val secondShape = second.shape

            val candidateIndex = broadcastCandidate(firstShape, secondShape)
            if (candidateIndex == 0) {
                return Pair(first, second)
            }

            return if (candidateIndex == 2) {
                Pair(
                    first,
                    second.broadcast(*firstShape)
                )
            } else {
                Pair(
                    first.broadcast(*secondShape),
                    second
                )
            }
        }

        private fun broadcastCandidate(firstShape: IntArray, secondShape: IntArray): Int {
            var candidateIndex = 0

            val (firstModifiedShape, secondModifiedShape) = if (firstShape.size > secondShape.size) {
                candidateIndex = 2

                Pair(firstShape,
                    IntArray(firstShape.size) {
                        if (it < firstShape.size - secondShape.size) {
                            1
                        } else {
                            secondShape[it - firstShape.size + secondShape.size]
                        }
                    })

            } else if (secondShape.size > firstShape.size) {
                candidateIndex = 1

                Pair(
                    IntArray(secondShape.size) {
                        if (it < secondShape.size - firstShape.size) {
                            1
                        } else {
                            firstShape[it - secondShape.size + firstShape.size]
                        }
                    },
                    secondShape
                )
            } else {
                Pair(firstShape, secondShape)
            }

            for ((firstDimension, secondDimension) in firstModifiedShape.zip(secondModifiedShape)) {
                if (firstDimension != secondDimension) {
                    if (firstDimension == 1) {
                        if (candidateIndex == 2) {
                            throw IllegalArgumentException(
                                "Cannot broadcast shape ${firstShape.joinToString(",", "[", "]")} " +
                                        "to ${secondShape.joinToString(",", "[", "]")}"
                            )
                        }
                        candidateIndex = 1
                    } else if (secondDimension == 1) {
                        if (candidateIndex == 1) {
                            throw IllegalArgumentException(
                                "Cannot broadcast shape ${secondShape.joinToString(",", "[", "]")} " +
                                        "to ${firstShape.joinToString(",", "[", "]")}"
                            )
                        }
                        candidateIndex = 2
                    } else {
                        if (candidateIndex <= 1) {
                            throw IllegalArgumentException(
                                "Cannot broadcast shape ${firstShape.joinToString(",", "[", "]")} " +
                                        "to ${secondShape.joinToString(",", "[", "]")}"
                            )
                        } else {
                            throw IllegalArgumentException(
                                "Cannot broadcast shape ${secondShape.joinToString(",", "[", "]")} " +
                                        "to ${firstShape.joinToString(",", "[", "]")}"
                            )
                        }
                    }
                }
            }

            return candidateIndex
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
    val result = FloatTensor(*other.shape)

    for (indexes in other.enumerateIndexes()) {
        result.set(indexes, this / other.get(indexes))
    }

    return result
}

fun Float.pow(tensor: FloatTensor): FloatTensor {
    val result = FloatTensor(*tensor.shape)

    for (indexes in tensor.enumerateIndexes()) {
        result.set(indexes, this.pow(tensor.get(indexes)))
    }

    return result
}
