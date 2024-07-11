package com.babylonml.tensor

import org.apache.commons.rng.UniformRandomProvider

class ByteTensor internal constructor(val shape: IntArray, private val data: Array<Any>) {
    constructor(shape: IntArray) : this(shape, createData(shape))

    internal fun enumerateIndexes(): Sequence<IntArray> {
        return doEnumerateIndexes(shape)
    }

    operator fun plus(other: FloatTensor): FloatTensor {
        val (first, second) = broadcast(this, other)
        return first.doAdd(second)
    }

    private fun doAdd(other: FloatTensor): FloatTensor {
        val result = FloatTensor(shape)
        for (indexes in enumerateIndexes()) {
            result.set(indexes, get(indexes) + other.get(indexes))
        }

        return result
    }

    internal fun get(indexes: IntArray): Byte {
        return doGet(indexes, data) as Byte
    }


    internal fun set(indexes: IntArray, value: Byte) {
        doSet(indexes, data, value)
    }


    val size = calculateSize(shape)

    fun broadcast(vararg newShape: Int, till: Int = shape.size): ByteTensor {
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

        for ((index, dimensions) in modifiedCurrentShape.zip(modifiedNewShape).withIndex().reversed()) {
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

        return ByteTensor(modifiedNewShape, modifiedData)
    }

    fun toFlatArray(): ByteArray {
        val result = ByteArray(size)

        for (indexes in enumerateIndexes()) {
            result[flattenIndex(indexes)] = get(indexes)
        }

        return result
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

        fun random(source: UniformRandomProvider, vararg shape: Int): ByteTensor {
            val result = ByteTensor(shape)

            val byteArray = ByteArray(1)
            for (indexes in result.enumerateIndexes()) {
                source.nextBytes(byteArray)
                result.set(indexes, byteArray[0])
            }

            return result
        }

        @Suppress("unused")
        fun natural(vararg shape: Int): ByteTensor {
            val result = ByteTensor(shape)
            var i = 0

            for (indexes in result.enumerateIndexes()) {
                result.set(indexes, i.toByte())
                i++
            }

            return result
        }

        @Suppress("unused")
        fun constant(value: Byte, vararg shape: Int): ByteTensor {
            val result = ByteTensor(shape)

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
                    0.toByte()
                }
            } else {
                Array(shape[0]) {
                    createData(shape.sliceArray(1 until shape.size))
                }
            }
        }

        private fun broadcast(first: ByteTensor, second: FloatTensor): Pair<ByteTensor, FloatTensor> {
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