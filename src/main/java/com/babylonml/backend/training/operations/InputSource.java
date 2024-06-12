package com.babylonml.backend.training.operations;

import org.jspecify.annotations.NonNull;

public interface InputSource {
    void addMiniBatchListener(@NonNull MiniBatchListener listener);
}
