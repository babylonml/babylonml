package com.babylonml.backend.common;

import it.unimi.dsi.fastutil.ints.IntImmutableList;

public record TensorPointer(long pointer, IntImmutableList shape, DType dtype) {
    public static TensorPointer NULL = new TensorPointer(0, IntImmutableList.of(), DType.F32);

    public enum DType {
        F32,
        INT8
    }
}


