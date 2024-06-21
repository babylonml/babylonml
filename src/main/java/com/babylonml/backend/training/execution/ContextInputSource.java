package com.babylonml.backend.training.execution;

import com.babylonml.backend.training.operations.MiniBatchListener;
import com.babylonml.backend.training.operations.Operation;
import org.jspecify.annotations.NonNull;

public interface ContextInputSource extends Operation {
    void addMiniBatchListener(@NonNull MiniBatchListener listener);
    int getSamplesCount();
    int gitLocalMiniBatchIndex();
    int getMiniBatchSize();
}
