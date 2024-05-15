package com.tornadoml.cpu;

import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.sampling.distribution.GaussianSampler;
import org.apache.commons.rng.sampling.distribution.ZigguratSampler;
import org.apache.commons.rng.simple.RandomSource;

public class LeakyLeRU implements ActivationFunction {
    private final long seed;

    public LeakyLeRU(long seed) {
        this.seed = seed;
    }

    @SuppressWarnings("unused")
    public LeakyLeRU() {
        this.seed = -1;
    }

    @Override
    public void value(float[] input, float[] result) {
        for (int i = 0; i < input.length; i++) {
            result[i] = input[i] > 0 ? input[i] : 0.01f * input[i];
        }
    }

    @Override
    public void derivative(float[] input, float[] result) {
        for (int i = 0; i < input.length; i++) {
            result[i] = input[i] > 0 ? 1.0f : 0.01f;
        }
    }

    @Override
    public void initWeighs(float[] weights, int inputSize) {
        initWeights(weights, inputSize, seed);
    }

    static void initWeights(float[] weights, int inputSize, long seed) {
        UniformRandomProvider source;
        if (seed != -1) {
            source = RandomSource.ISAAC.create(seed);
        } else {
            source = RandomSource.ISAAC.create();
        }

        var sampler = GaussianSampler.of(
                ZigguratSampler.NormalizedGaussian.of(source),
                0.0, Math.sqrt(2.0 / inputSize));

        var samples = sampler.samples(weights.length).iterator();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float) samples.nextDouble();
        }

    }
}
