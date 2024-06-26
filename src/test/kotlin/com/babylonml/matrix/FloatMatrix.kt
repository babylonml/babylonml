package com.babylonml.matrix

import com.babylonml.backend.training.optimizer.GradientOptimizer
import com.babylonml.backend.training.execution.TrainingExecutionContext
import com.babylonml.backend.training.operations.Variable
import com.babylonml.FloatVector
import com.babylonml.backend.training.operations.Tensor
import org.apache.commons.rng.UniformRandomProvider
import kotlin.math.sqrt

class FloatMatrix(val rows: Int, val cols: Int) {
    companion object {
        fun random(rows: Int, cols: Int, source: UniformRandomProvider): FloatMatrix {
            val result = FloatMatrix(rows, cols)
            result.fillRandom(source)
            return result
        }

        fun random(rows: Int, cols: Int, origin: Float, boundary: Float, source: UniformRandomProvider): FloatMatrix {
            val result = FloatMatrix(rows, cols)
            result.fillRandom(source, origin, boundary)
            return result
        }
    }

    constructor(rows: Int, cols: Int, data: FloatArray) : this(rows, cols) {
        assert(data.size == rows * cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                this.data[i][j] = data[i * cols + j]
            }
        }
    }

    constructor(data: Array<FloatArray>) : this(data.size, data[0].size) {
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                this.data[i][j] = data[i][j]
            }
        }
    }

    internal val data = Array(rows) { FloatArray(cols) }

    val size = rows * cols

    operator fun times(other: FloatMatrix): FloatMatrix {
        assert(cols == other.rows)

        val result = FloatMatrix(rows, other.cols)

        for (i in 0 until rows) {
            for (j in 0 until other.cols) {
                for (k in 0 until cols) {
                    result.data[i][j] += data[i][k] * other.data[k][j]
                }
            }
        }

        return result
    }

    operator fun plus(other: FloatMatrix): FloatMatrix {
        assert(rows == other.rows && cols == other.cols)

        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i][j] + other.data[i][j]
            }
        }

        return result
    }

    operator fun plus(float: Float): FloatMatrix {
        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i][j] + float
            }
        }

        return result
    }

    operator fun minus(other: FloatMatrix): FloatMatrix {
        assert(rows == other.rows && cols == other.cols)

        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i][j] - other.data[i][j]
            }
        }

        return result
    }

    fun hadamardMul(other: FloatMatrix): FloatMatrix {
        assert(rows == other.rows && cols == other.cols)

        val result = FloatMatrix(rows, cols)
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i][j] * other.data[i][j]
            }
        }

        return result
    }

    fun dotMulRows(other: FloatMatrix): FloatMatrix {
        assert(rows == other.rows && cols == other.cols)
        val result = FloatMatrix(1, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[0][j] += data[i][j] * other.data[i][j]
            }
        }

        return result
    }

    fun dotMulCols(other: FloatMatrix): FloatMatrix {
        assert(rows == other.rows && cols == other.cols)
        val result = FloatMatrix(rows, 1)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][0] += data[i][j] * other.data[i][j]
            }
        }

        return result
    }

    fun toVariable(exec: TrainingExecutionContext, optimizer: GradientOptimizer, learningRate: Float): Variable {
        return Variable(exec, optimizer, toTensor(), learningRate)
    }

    fun toVariable(
        name: String, exec: TrainingExecutionContext, optimizer: GradientOptimizer,
        learningRate: Float
    ): Variable {
        return Variable(name, exec, optimizer, toTensor(), learningRate)
    }

    fun toTensor(dimensions: Int = 2): Tensor {
        if (dimensions < 2) {
            throw IllegalArgumentException("Tensor dimensions must be at least 2")
        }

        val shape = IntArray(dimensions) {
            1
        }

        shape[shape.size - 2] = rows
        shape[shape.size - 1] = cols

        return Tensor(toFlatArray(), shape)
    }

    operator fun times(float: Float): FloatMatrix {
        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i][j] * float
            }
        }

        return result
    }

    operator fun times(int: Int): FloatMatrix {
        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i][j] * int
            }
        }

        return result
    }

    operator fun div(float: Float): FloatMatrix {
        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i][j] / float
            }
        }

        return result
    }

    operator fun div(matrix: FloatMatrix): FloatMatrix {
        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i][j] / matrix.data[i][j]
            }
        }

        return result
    }

    operator fun div(int: Int): FloatMatrix {
        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i][j] / int
            }
        }

        return result
    }

    @Suppress("unused")
    fun broadcastByColumns(cols: Int): FloatMatrix {
        if (this.cols != 1) {
            throw IllegalArgumentException("Matrix must have only one column")
        }

        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i][0]
            }
        }

        return result
    }

    fun broadcastByRows(rows: Int): FloatMatrix {
        if (this.rows != 1) {
            throw IllegalArgumentException("Matrix must have only one row")
        }

        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[0][j]
            }
        }

        return result
    }

    fun subColumns(start: Int, count: Int): FloatMatrix {
        val result = FloatMatrix(rows, count)

        for (i in 0 until rows) {
            for (j in 0 until count) {
                result.data[i][j] = data[i][start + j]
            }
        }

        return result
    }

    fun subRows(start: Int, count: Int): FloatMatrix {
        val result = FloatMatrix(count, cols)

        for (i in 0 until count) {
            for (j in 0 until cols) {
                result.data[i][j] = data[start + i][j]
            }
        }

        return result
    }

    fun sqrt(): FloatMatrix {
        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = sqrt(data[i][j])
            }
        }

        return result
    }

    fun ln(): FloatMatrix {
        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = kotlin.math.ln(data[i][j])
            }
        }

        return result
    }

    fun max(matrix: FloatMatrix): FloatMatrix {
        assert(rows == matrix.rows && cols == matrix.cols)

        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = if (data[i][j] > matrix.data[i][j]) data[i][j] else matrix.data[i][j]
            }
        }

        return result
    }

    fun exp(): FloatMatrix {
        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = kotlin.math.exp(data[i][j])
            }
        }

        return result
    }

    fun softMaxByColumns(): FloatMatrix {
        val exp = exp()
        val sum = exp.transpose().reduceByColumns().broadcastRows(rows)
        return exp / sum
    }

    fun softMaxByRows(): FloatMatrix {
        val exp = exp()
        val sum = exp.reduceByColumns().broadcastColumns(cols)
        return exp / sum
    }


    fun fillRandom(source: UniformRandomProvider) {
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                data[i][j] = source.nextFloat(-1.0f, 1.0f)
            }
        }
    }

    fun fillRandom(source: UniformRandomProvider, origin: Float, boundary: Float) {
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                data[i][j] = source.nextFloat(origin, boundary)
            }
        }
    }

    fun transpose(): FloatMatrix {
        val result = FloatMatrix(cols, rows)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[j][i] = data[i][j]
            }
        }

        return result
    }

    fun reduceByColumns(): FloatVector {
        val result = FloatVector(rows)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i] += data[i][j]
            }
        }

        return result
    }

    @Suppress("unused")
    fun sumByColumns(): FloatMatrix {
        val result = FloatMatrix(rows, 1)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][0] += data[i][j]
            }
        }

        return result
    }

    fun sumByRows(): FloatMatrix {
        val result = FloatMatrix(1, cols)

        for (i in 0 until cols) {
            for (j in 0 until rows) {
                result.data[0][i] += data[j][i]
            }
        }

        return result
    }

    fun reduceByRows(): FloatVector {
        val result = FloatVector(cols)

        for (i in 0 until cols) {
            for (j in 0 until rows) {
                result.data[i] += data[j][i]
            }
        }

        return result
    }

    fun toFlatArray(): FloatArray {
        val result = FloatArray(rows * cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result[i * cols + j] = data[i][j]
            }
        }

        return result
    }

    operator fun set(i: Int, j: Int, value: Float) {
        data[i][j] = value
    }

    override fun toString(): String {
        return "FloatMatrix(rows: $rows, cols: $cols)"
    }
}

operator fun Int.times(other: FloatMatrix): FloatMatrix {
    return other * this
}

operator fun Float.times(other: FloatMatrix): FloatMatrix {
    return other * this
}

operator fun Float.div(other: FloatMatrix): FloatMatrix {
    val result = FloatMatrix(other.rows, other.cols)

    for (i in 0 until other.rows) {
        for (j in 0 until other.cols) {
            result.data[i][j] = this / other.data[i][j]
        }
    }

    return result
}
