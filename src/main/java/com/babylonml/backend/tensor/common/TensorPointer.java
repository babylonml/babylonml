package com.babylonml.backend.tensor.common;

import it.unimi.dsi.fastutil.ints.IntImmutableList;

public record TensorPointer(long pointer, IntImmutableList shape, DType dtype, MemoryKind memoryKind) {
    @SuppressWarnings("unused")
    public static TensorPointer NULL = new TensorPointer(0, IntImmutableList.of(), DType.NONE, MemoryKind.NONE);

    public enum DType {
        F32,
        F16,
        INT32,
        INT8,
        NONE
    }

    public enum MemoryKind {
        SINGLE_PASS,
        OPERATION_LOCAL,
        RESIDENT,
        INPUT,
        NONE
    }
}


