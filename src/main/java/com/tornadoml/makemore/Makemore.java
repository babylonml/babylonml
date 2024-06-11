package com.tornadoml.makemore;

import com.tornadoml.cpu.CategoricalCrossEntropyFunction;
import com.tornadoml.cpu.DenseLayer;
import com.tornadoml.cpu.EmbeddingLayer;
import com.tornadoml.cpu.GeLU;
import com.tornadoml.cpu.MultinomialSampler;
import com.tornadoml.cpu.NeuralNetwork;
import com.tornadoml.cpu.SoftMaxLayer;
import com.tornadoml.cpu.WeightsOptimizer.OptimizerType;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public class Makemore {

  private static final int BLOCK_SIZE = 3;
  private static final int EMBEDDINGS_DIM = 10;

  private static Dataset smallSet;
  private static Dataset trainingSet;
  private static Dataset testSet;
  private static NeuralNetwork network;

  public static void main(String[] args) throws Exception {
    final var targetSize = 27;

    final var words = loadNames();
    smallSet = buildDataset(words, 0, 5);
    smallSet.prettyPrint();

    final var shuffledWords = new ArrayList<>(words);
    Collections.shuffle(shuffledWords);

    final var s = words.size() * 9 / 10;

    trainingSet = buildDataset(words, 0, s);
    testSet = buildDataset(words, s, words.size());

    network = new NeuralNetwork(
        new CategoricalCrossEntropyFunction(),
        new EmbeddingLayer(27, EMBEDDINGS_DIM, BLOCK_SIZE),
        new DenseLayer(BLOCK_SIZE * EMBEDDINGS_DIM, 100, new GeLU(), OptimizerType.SIMPLE),
        new DenseLayer(100, 27, new GeLU(), OptimizerType.SIMPLE),
        new SoftMaxLayer(27)
    );

    network.fit(
        trainingSet.x,
        trainingSet.y,
        BLOCK_SIZE,
        targetSize,
        trainingSet.x.length,
        1000,
        0.01f,
        -1,
        false
    );

    generateNames(20);

    final var trainAccuracy = network.test(trainingSet.x, trainingSet.y);
    System.out.println("Training accuracy: " + trainAccuracy);

    final var testAccuracy = network.test(testSet.x, testSet.y);
    System.out.println("Test accuracy: " + testAccuracy);
  }

  private static @NotNull List<String> loadNames() throws IOException {
    final var namesRes = Thread.currentThread().getContextClassLoader().getResource("makemore/names.txt");
    Objects.requireNonNull(namesRes, "names.txt not found");
    final var names = Files.readAllLines(Paths.get(namesRes.getPath()));

    System.out.printf("Loaded %d names\n", names.size());
    return names;
  }

  private static Dataset buildDataset(List<String> words, int from, int to) {

    List<float[]> x = new ArrayList<>();
    List<float[]> y = new ArrayList<>();

    for (int i = from; i < to; i++) {
      final var chars = words.get(i).toCharArray();

      var context = new float[BLOCK_SIZE];

      for (var c : chars) {
        final var ix = charToInt(c);
        x.add(context);
        y.add(oneHot(ix));

        final var newContext = new float[BLOCK_SIZE];
        System.arraycopy(context, 1, newContext, 0, BLOCK_SIZE - 1);
        newContext[BLOCK_SIZE - 1] = ix;
        context = newContext;
      }

      x.add(context);
      y.add(oneHot(0));
    }

    System.out.printf("Built dataset from %d to %d, of total size %d\n", from, to, x.size());

    return new Dataset(x.toArray(new float[0][]), y.toArray(new float[0][]));
  }

  private static void generateNames(int n) {
    final var sampler = new MultinomialSampler();
    for (int i = 0; i < n; i++) {

      boolean done = false;
      var context = new float[BLOCK_SIZE];
      var word = new ArrayList<Character>();
      while (!done) {

        final var prediction = network.predict(context);
        final var nextChar = sampler.sampleOne(prediction);

        if (nextChar == 0) {
          done = true;
        } else {
          word.add(intToChar(nextChar));

          final var newContext = new float[BLOCK_SIZE];
          System.arraycopy(context, 1, newContext, 0, BLOCK_SIZE - 1);
          newContext[BLOCK_SIZE - 1] = nextChar;
          context = newContext;
        }
      }

      System.out.printf("Generated name: %s\n", new String(word.stream().mapToInt(c -> c).toArray(), 0, word.size()));
    }
  }

  record Dataset(float[][] x, float[][] y) {

    public void prettyPrint() {
      prettyPrint(x.length);
    }

    public void prettyPrint(int maxLength) {
      final var n = Math.min(x.length, maxLength);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < x[i].length; j++) {
          System.out.print(intToChar(((int) x[i][j])));
        }
        System.out.print(" -> ");
        System.out.println(intToChar(maxIndex(y[i])));
      }
    }
  }

  private static char intToChar(int i) {
    if (i == 0) {
      return '.';
    } else if (i >= 1 && i <= 26) {
      return (char) ('a' + i - 1);
    } else {
      throw new IllegalArgumentException("" + i);
    }
  }

  private static int charToInt(char c) {
    if (c == '.') {
      return 0;
    } else if (c >= 'a' && c <= 'z') {
      return c - 'a' + 1;
    } else {
      throw new IllegalArgumentException("" + c);
    }
  }

  private static float[] oneHot(int i) {
    var result = new float[27];
    result[i] = 1f;
    return result;
  }

  private static int maxIndex(float[] arr) {
    var max = arr[0];
    var maxIndex = 0;
    for (int i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
        max = arr[i];
        maxIndex = i;
      }
    }
    return maxIndex;
  }
}
