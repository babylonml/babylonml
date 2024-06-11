package com.tornadoml.cpu;

import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;

public class MultinomialSampler {

  private final UniformRandomProvider source = RandomSource.ISAAC.create();

  public int sampleOne(float[] probabilities) {
    final var rand = source.nextFloat();
    var sum = 0.0f;
    int i;
    for (i = 0; i < probabilities.length && sum <= rand; i++) {
      sum += probabilities[i];
    }

    return i - 1;
  }
}
