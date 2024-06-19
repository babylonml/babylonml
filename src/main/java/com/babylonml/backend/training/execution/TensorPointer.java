package com.babylonml.backend.training.execution;

public record TensorPointer(long pointer, int[] shape, TrainingExecutionContext executionContext) {
    public float[] buffer() {
        assert executionContext != null;
        return executionContext.getMemoryBuffer(pointer);
    }

    public int offset() {
        return TrainingExecutionContext.addressOffset(pointer);
    }
}
