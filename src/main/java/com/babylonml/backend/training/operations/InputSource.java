package com.babylonml.backend.training.operations;

public interface InputSource {
    void addMiniBatchListener(MiniBatchListener listener);
}
