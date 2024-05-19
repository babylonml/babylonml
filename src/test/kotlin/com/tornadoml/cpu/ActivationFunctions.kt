package com.tornadoml.cpu


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