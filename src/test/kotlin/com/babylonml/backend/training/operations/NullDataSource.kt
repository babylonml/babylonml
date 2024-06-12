package com.babylonml.backend.training.operations

class NullDataSource : InputSource {
    override fun addMiniBatchListener(listener: MiniBatchListener) {
        // Do nothing
    }
}