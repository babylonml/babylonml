package com.babylonml.backend.training.operations;

public interface MiniBatchListener {
    void onMiniBatchStart(long miniBatchIndex, int miniBatchSize);
}
