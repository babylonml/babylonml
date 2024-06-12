package com.babylonml

import com.babylonml.matrix.FloatMatrix
import org.apache.commons.rng.UniformRandomProvider
import kotlin.math.exp
import kotlin.math.sqrt

class FloatVector(val size: Int) {
    internal val data = FloatArray(size)

    operator fun plus(other: FloatVector): FloatVector {
        assert(size == other.size)
        val result = FloatVector(size)

        for (i in 0 until size) {
            result.data[i] = data[i] + other.data[i]
        }

        return result
    }

    operator fun times(value: Float): FloatVector {
        val result = FloatVector(size)

        for (i in 0 until size) {
            result.data[i] = data[i] * value
        }

        return result
    }

    operator fun times(other: FloatVector): FloatVector {
        assert(size == other.size)
        val result = FloatVector(size)

        for (i in 0 until size) {
            result.data[i] = data[i] * other.data[i]
        }

        return result
    }

    operator fun div(value: Float): FloatVector {
        val result = FloatVector(size)

        for (i in 0 until size) {
            result.data[i] = data[i] / value
        }

        return result
    }

    operator fun plus(float: Float): FloatVector {
        val result = FloatVector(size)

        for (i in 0 until size) {
            result.data[i] = data[i] + float
        }

        return result
    }

    operator fun minus(other: FloatVector): FloatVector {
        assert(size == other.size)
        val result = FloatVector(size)

        for (i in 0 until size) {
            result.data[i] = data[i] - other.data[i]
        }

        return result
    }

    operator fun div(int: Int): FloatVector {
        val result = FloatVector(size)

        for (i in 0 until size) {
            result.data[i] = data[i] / int
        }

        return result
    }

    fun broadcastColumns(cols: Int): FloatMatrix {
        val result = FloatMatrix(size, cols)

        for (i in 0 until size) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i]
            }
        }

        return result
    }

    fun broadcastRows(rows: Int): FloatMatrix {
        val result = FloatMatrix(rows, size)

        for (i in 0 until rows) {
            for (j in 0 until size) {
                result.data[i][j] = data[j]
            }
        }

        return result
    }

    fun sqrt(): FloatVector {
        val result = FloatVector(size)

        for (i in 0 until size) {
            result.data[i] = sqrt(data[i])
        }

        return result
    }

    fun exp(): FloatVector {
        val result = FloatVector(size)

        for (i in 0 until size) {
            result.data[i] = exp(data[i])
        }

        return result
    }

    fun max(vector: FloatVector): FloatVector {
        assert(size == vector.size)
        val result = FloatVector(size)

        for (i in 0 until size) {
            result.data[i] = if (data[i] > vector.data[i]) data[i] else vector.data[i]
        }

        return result
    }

    fun sum(): Float {
        var sum = 0.0f
        for (i in 0 until size) {
            sum += data[i]
        }
        return sum
    }

    fun fillRandom(source: UniformRandomProvider) {
        for (i in 0 until size) {
            data[i] = source.nextFloat()
        }
    }

    fun toArray() = data.copyOf()

    override fun toString(): String {
        return "FloatVector(size=$size)"
    }
}

operator fun Float.times(vector: FloatVector) = vector * this

operator fun Float.div(vector: FloatVector): FloatVector {
    val result = FloatVector(vector.size)

    for (i in 0 until vector.size) {
        result.data[i] = this / vector.data[i]
    }

    return result

}
