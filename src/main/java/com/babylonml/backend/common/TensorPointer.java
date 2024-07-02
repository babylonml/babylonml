package com.babylonml.backend.common;

import it.unimi.dsi.fastutil.ints.IntImmutableList;

public record TensorPointer(long pointer, IntImmutableList shape){
    public static TensorPointer NULL = new TensorPointer(0, IntImmutableList.of());
}
