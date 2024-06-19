package com.babylonml.backend.training.initializer;

import com.babylonml.backend.cpu.TensorOperations;

import java.util.Arrays;

public final class ZeroInitializer implements Initializer {
    @Override
    public void initialize(float[] matrix, int offset, int[] shape) {
        Arrays.fill(matrix, offset, offset + TensorOperations.stride(shape), 0.0f);
    }
}
