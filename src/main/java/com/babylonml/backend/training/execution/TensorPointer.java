package com.babylonml.backend.training.execution;

import org.jspecify.annotations.Nullable;

import java.util.Objects;

public record TensorPointer(long pointer, int[] shape, @Nullable TrainingExecutionContext executionContext) {
    public float[] buffer() {
        Objects.requireNonNull(executionContext, "Execution context is not set");
        return executionContext.getMemoryBuffer(pointer);
    }

    public int offset() {
        return TrainingExecutionContext.addressOffset(pointer);
    }
}
