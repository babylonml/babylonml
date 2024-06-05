package com.tornadoml.cpu

/**
 * Samples and actual results are grouped by columns
 */
fun mseCostFunctionByColumns(actual: FloatMatrix, expected: FloatMatrix): Float {
    val diff = expected - actual
    val squared = diff.dotMulRows(diff)

    assert(squared.rows == 1)

    return squared.reduceByColumns().sum() / squared.cols
}

fun mseCostFunctionByRows(actual: FloatMatrix, expected: FloatMatrix): Float {
    val diff = expected - actual
    val squared = diff.dotMulCols(diff)

    assert(squared.cols == 1)

    return squared.reduceByRows().sum() / squared.rows
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

    return -sum / actual.cols
}