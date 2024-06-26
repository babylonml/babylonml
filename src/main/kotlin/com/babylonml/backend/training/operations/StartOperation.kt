package com.babylonml.backend.training.operations

interface StartOperation : Operation {
    fun calculateGradientUpdate()
}
