package com.tornadoml.cpu

import org.apache.commons.rng.UniformRandomProvider
import kotlin.math.sqrt

class FloatMatrix(val rows: Int, val cols: Int) {
    constructor(rows: Int, cols: Int, data: FloatArray) : this(rows, cols) {
        assert(data.size == rows * cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                this.data[i][j] = data[i * cols + j]
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

    fun dotMul(other: FloatMatrix): FloatMatrix {
        assert(rows == other.rows && cols == other.cols)
        val result = FloatMatrix(1, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[0][j] += data[i][j] * other.data[i][j]
            }
        }

        return result
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

    operator fun div(int: Int): FloatMatrix {
        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i][j] / int
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


    fun fillRandom(source: UniformRandomProvider) {
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                data[i][j] = source.nextFloat()
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

    fun reduce(): FloatVector {
        val result = FloatVector(rows)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i] += data[i][j]
            }
        }

        return result
    }

    fun copy(): FloatMatrix {
        val result = FloatMatrix(rows, cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result.data[i][j] = data[i][j]
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

    fun toArray() = data.clone()

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