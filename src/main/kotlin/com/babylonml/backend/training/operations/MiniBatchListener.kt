package com.babylonml.backend.training.operations

interface MiniBatchListener {
    fun onMiniBatchStart(miniBatchIndex: Long, miniBatchSize: Int)
}
