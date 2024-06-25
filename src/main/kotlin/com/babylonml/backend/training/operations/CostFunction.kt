package com.babylonml.backend.training.operations

interface CostFunction : Operation {
    fun trainingMode()
    fun fullPassCalculationMode()
}
