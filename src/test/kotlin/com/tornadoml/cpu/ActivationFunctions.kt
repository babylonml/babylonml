package com.tornadoml.cpu

import kotlin.math.sqrt
import kotlin.math.tanh


fun leakyLeRU(x: FloatMatrix, alpha: Float): FloatMatrix {
    val result = FloatMatrix(x.rows, x.cols)

    for (i in 0 until x.rows) {
        for (j in 0 until x.cols) {
            result.data[i][j] = if (x.data[i][j] > 0) x.data[i][j] else alpha * x.data[i][j]
        }
    }

    return result
}

fun leakyLeRUDerivative(x: FloatMatrix, alpha: Float): FloatMatrix {
    val result = FloatMatrix(x.rows, x.cols)
    for (i in 0 until x.rows) {
        for (j in 0 until x.cols) {
            result.data[i][j] = if (x.data[i][j] > 0) 1.0f else alpha
        }
    }

    return result
}

fun  geLU(x: FloatMatrix): FloatMatrix {
    val result = FloatMatrix(x.rows, x.cols)

    for (i in 0 until x.rows) {
        for (j in 0 until x.cols) {
            //0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x^3)))
            val xij = x.data[i][j]
            result.data[i][j] =
                0.5f * xij * (1 + tanh(sqrt(2 / Math.PI) * (xij + 0.044715f * xij * xij * xij)).toFloat())
        }
    }

    return result
}