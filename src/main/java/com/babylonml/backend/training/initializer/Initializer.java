package com.babylonml.backend.training.initializer;

public interface Initializer {
    void initialize(float[] matrix, int offset, int[] shape);

    @SuppressWarnings("unused")
    static Initializer zero() {
        return new ZeroInitializer();
    }

    static Initializer he(long seed) {
        return new HeInitializer(seed);
    }

    static Initializer he() {
        return he(0x213aa3e1e3d5e7e1L);
    }
}
