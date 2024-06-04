package com.tornadoml.cpu

/**
 * Samples and actual results are grouped by columns
 */
fun mseCostFunction(actual: FloatMatrix, expected: FloatMatrix): Float {
    val diff = expected - actual
    val squared = diff.dotMul(diff)

    assert(squared.rows == 1)

    return squared.reduceByColumns().sum() / squared.cols
}

/**
 * Samples and actual results are grouped by columns
 */
fun mseCostFunctionDerivative(actual: FloatMatrix, expected: FloatMatrix): FloatMatrix {
    return actual - expected
}

fun crossEntropyByRows(actual: FloatMatrix, expected: FloatMatrix): Float {
    val logActual = actual.ln()
    val mul = expected.dotMul(logActual)

    val sum = mul.reduceByColumns().sum()

    return -sum / actual.cols
}