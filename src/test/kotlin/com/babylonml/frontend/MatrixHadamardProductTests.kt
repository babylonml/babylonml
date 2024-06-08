package com.babylonml.frontend

import com.babylonml.backend.training.operations.HadamardProduct

class MatrixHadamardProductTests : BroadcastableBinaryOperationTestsSuite<HadamardProduct>(
    HadamardProduct::class,
    { a, b -> a.hadamardMul(b) },
    { MatrixDims(it.rows, it.columns) }
)

