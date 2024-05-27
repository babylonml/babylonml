package com.tornadoml.cpu;

import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;

/**
 * Layer that maps integers to embedding vectors.
 */
public class EmbeddingLayer implements TrainableLayer {
  private final float[][] lookupTable;
  private final int inputSize;
  private final int outputSize;
  private final int dimensionality;
  private final int vocabularySize;

  private float[][] bestLookupTable;


  public EmbeddingLayer(int vocabularySize, int dimensionality, int inputSize) {
    this(vocabularySize, dimensionality, inputSize, -1);
  }

  public EmbeddingLayer(int vocabularySize, int dimensionality, int inputSize, long seed) {
    this.dimensionality = dimensionality;
    this.vocabularySize = vocabularySize;
    this.lookupTable = new float[vocabularySize][dimensionality];

    this.inputSize = inputSize;
    this.outputSize = inputSize * dimensionality;

    initEmbeddings(vocabularySize, seed);
  }

  private void initEmbeddings(int vocabularySize, long seed) {
    final UniformRandomProvider source;
    if (seed != -1) {
      source = RandomSource.ISAAC.create(seed);
    } else {
      source = RandomSource.ISAAC.create();
    }

    for (int i = 0; i < vocabularySize; i++) {
      for (int j = 0; j < this.dimensionality; j++) {
        lookupTable[i][j] = source.nextFloat(-0.5f, 0.5f);
      }
    }
  }

  @Override
  public void predict(float[] input, float[] prediction, int batchSize) {
    forwardTraining(input, 0, null, prediction, batchSize);
  }

  @Override
  public void forwardTraining(float[] input, int inputOffset, float[] activationArgument, float[] prediction, int batchSize) {

    for (int b = 0; b < batchSize; b++) {

      for (int i = 0; i < inputSize; i++) {

        // todo: float to int casting concerns me
        final var embedding = lookupTable[(int) input[inputOffset + batchSize * i + b]];

        for (int d = 0; d < dimensionality; d++) {
          final var destOffset = batchSize * (i * dimensionality + d) + b;
          prediction[destOffset] = embedding[d];
          if (activationArgument != null) {
            activationArgument[destOffset] = embedding[d];
          }
        }
      }
    }
  }

  @Override
  public void updateWeightsAndBiases(float[] weightsDelta, float[] biasesDelta, float learningRate) {
    for (int i = 0; i < vocabularySize; i++) {
      for (int j = 0; j < dimensionality; j++) {
        lookupTable[i][j] -= learningRate * weightsDelta[i * dimensionality + j];
      }
    }
  }

  @Override
  public void saveWeightsAndBiases() {
    this.bestLookupTable = lookupTable.clone();
  }

  @Override
  public void restoreBestWeightsAndBiases() {
    // todo: verify this
    for (int i = 0; i < vocabularySize; i++) {
      System.arraycopy(bestLookupTable[i], 0, lookupTable[i], 0, dimensionality);
    }
  }


  @Override
  public void backwardLastLayer(float[] input, float[] previousLayerActivationArgument, float[] currentLayerActivationArgument, float[] currentLayerErrors, float[] previousLayerErrors, float[] calculatedWeightsDelta, float[] calculatedBiasesDelta, int miniBatchSize) {
    throw new UnsupportedOperationException("Embedding layer can be used only as the first layer in the network.");
  }

  @Override
  public void backwardSingleLayerNoError(float[] input, float[] currentLayerActivationArgument, float[] costFunctionDerivative, float[] calculatedWeightsDelta, float[] calculatedBiasesDelta, int miniBatchSize) {
    throw new UnsupportedOperationException("Embedding layer can be used only as the first layer in the network.");
  }

  @Override
  public void backwardMiddleLayer(float[] input, float[] errors, float[] previousLayerActivationArgument, float[] previousLayerErrors, float[] weightsDelta, float[] biasesDelta, int miniBatchSize) {
    throw new UnsupportedOperationException("Embedding layer can be used only as the first layer in the network.");
  }

  @Override
  public void backwardFirstLayer(float[] input, int inputOffset, float[] currentLayerErrorsInput, float[] weightsGradientOutput, float[] biasesGradientOutput, int batchSize) {

    for (int b = 0; b < batchSize; b++) {

      for (int i = 0; i < inputSize; i++) {
        final int in = (int) input[inputOffset + i * batchSize + b];

        for (int d = 0; d < dimensionality; d++) {
          final var gradOffset = in * dimensionality + d;
          weightsGradientOutput[gradOffset] += currentLayerErrorsInput[batchSize * (i * dimensionality + d) + b];
        }
      }
    }
  }


  @Override
  public int getInputSize() {
    return inputSize;
  }

  @Override
  public int getOutputSize() {
    return outputSize;
  }

  @Override
  public int getWeightsSize() {
    return vocabularySize * dimensionality;
  }

  public float[][] getLookupTable() {
    return lookupTable;
  }
}
