package com.tornadoml.cpu;

@SuppressWarnings("unused")
public final class GeLU implements ActivationFunction {
    private static final double SCALAR_1 = 0.5f;
    private static final double SCALAR_2 = 1.0f;
    private static final double SCALAR_3 = Math.sqrt(2 / Math.PI);
    private static final double SCALAR_4 = 0.044715f;
    private static final double SCALAR_5 = 3 * SCALAR_4;

    private final long seed;

    @SuppressWarnings("unused")
    public GeLU(long seed) {
        this.seed = seed;
    }

    public GeLU() {
        this.seed = -1;
    }

    @Override
    public void value(float[] input, float[] result, int length) {
        assert input.length >= length;
        assert result.length >= length;

        for (int i = 0; i < length; i++) {
            // GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x^3)))
            result[i] =
                    (float) (input[i] * SCALAR_1 * (SCALAR_2 +
                            Math.tanh(SCALAR_3 *
                                    (input[i] + SCALAR_4 * input[i] * input[i] * input[i]))));
        }
    }

    @Override
    public void derivative(float[] input, float[] result, int length) {
        assert input.length >= length;
        assert result.length >= length;

        for (int i = 0; i < length; i++) {
            var tanh = (float)
                    Math.tanh(SCALAR_3 * (input[i] + SCALAR_4 * input[i] * input[i] * input[i]));
            // d(GeLU(x))/dx = 0.5 * (1 + tanh + x * (1 - tanh^2) * (sqrt(2 / PI) + 3 * 0.044715 * x^2))
            result[i] = (float)
                    (SCALAR_1 *
                            (SCALAR_2 + tanh + input[i] * (SCALAR_3 +
                                    SCALAR_5 * input[i] * input[i]) * (SCALAR_2 - tanh * tanh)));
        }
    }

    @Override
    public void initWeighs(float[] weights, int inputSize) {
        LeakyLeRU.initWeights(weights, inputSize, seed);
    }
}
