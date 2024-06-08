package com.babylonml.frontend

import com.babylonml.backend.training.operations.Add

class MatrixAdditionTests : BroadcastableBinaryOperationTestsSuite<Add>(
    Add::class,
    { a, b -> a + b },
    { MatrixDims(it.rows, it.columns) }
)

