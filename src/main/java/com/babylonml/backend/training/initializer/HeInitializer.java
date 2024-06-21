package com.babylonml.backend.training.initializer;

import com.babylonml.backend.cpu.TensorOperations;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.sampling.distribution.GaussianSampler;
import org.apache.commons.rng.sampling.distribution.ZigguratSampler;
import org.apache.commons.rng.simple.RandomSource;

public final class HeInitializer implements Initializer {
    private final long seed;

    public HeInitializer(long seed) {
        this.seed = seed;
    }

    @Override
    public void initialize(float[] matrix, int offset, int[] shape) {
        UniformRandomProvider source;

        if (seed != -1) {
            source = RandomSource.ISAAC.create(seed);
        } else {
            source = RandomSource.ISAAC.create();
        }

        var sampler = GaussianSampler.of(
                ZigguratSampler.NormalizedGaussian.of(source),
                0.0, Math.sqrt(2.0 / shape[0]));

        var size = TensorOperations.stride(shape);
        var samples = sampler.samples(size).iterator();

        for (int i = 0; i < size; i++) {
            matrix[offset + i] = (float) samples.nextDouble();
        }
    }
}

