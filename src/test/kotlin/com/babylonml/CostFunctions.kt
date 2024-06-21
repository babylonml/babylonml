package com.babylonml

import com.babylonml.matrix.FloatMatrix

fun mseCostFunction(actual: FloatMatrix, expected: FloatMatrix): Float {
    val diff = expected - actual
    val squared = diff.dotMulCols(diff)

    assert(squared.cols == 1)

    return squared.reduceByRows().sum()
}

/**
 * Samples and actual results are grouped by columns
 */
fun mseCostFunctionDerivative(actual: FloatMatrix, expected: FloatMatrix): FloatMatrix {
    return actual - expected
}

fun crossEntropyByRows(actual: FloatMatrix, expected: FloatMatrix): Float {
    val logActual = actual.ln()
    val mul = expected.dotMulRows(logActual)

    val sum = mul.reduceByColumns().sum()

    return -sum / actual.rows
}