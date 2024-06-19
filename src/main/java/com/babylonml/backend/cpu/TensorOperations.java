package com.babylonml.backend.cpu;


import org.jspecify.annotations.NonNull;

import java.util.Arrays;

public final class TensorOperations {

    public static int stride(int[] shape) {
        int stride = 1;
        for (int dim : shape) {
            stride *= dim;
        }
        return stride;
    }

    public static int @NonNull [] calculateMaxShape(int[] leftShape, int[] rightShape) {
        final int[] maxShape;

        if (leftShape.length == rightShape.length) {
            maxShape = new int[leftShape.length];
            for (int i = 0; i < leftShape.length; i++) {
                maxShape[i] = Math.max(leftShape[i], rightShape[i]);
            }
        } else if (leftShape.length < rightShape.length) {
            maxShape = rightShape;
        } else {
            maxShape = leftShape;
        }

        return maxShape;
    }

    private static boolean isNotBroadcastCompatible(int[] firstShape, int[] secondShape) {
        for (int i = 0; i < firstShape.length; i++) {
            if (firstShape[i] != secondShape[i] && firstShape[i] != 1 && secondShape[i] != 1) {
                return true;
            }
        }

        return false;
    }

    /**
     * Check if two shapes are broadcast compatible and return the candidate for broadcasting.
     *
     * @param firstShape  first shape
     * @param secondShape second shape
     * @return 0 if shapes are not needed to be broadcast, 1 if first shape is a candidate for broadcasting
     * and 2 if second shape is a candidate for broadcasting. -1 if shapes are not broadcast compatible.
     */
    public static int broadcastCandidate(int[] firstShape, int[] secondShape) {
        int candidate = 0;

        if (firstShape.length < secondShape.length) {
            candidate = 1;
        } else if (firstShape.length > secondShape.length) {
            candidate = 2;
        }

        var size = Math.min(firstShape.length, secondShape.length);

        for (int i = 0; i < size; i++) {
            if (firstShape[i] != secondShape[i]) {
                if (firstShape[i] == 1) {
                    if (candidate == 2) {
                        return -1;
                    }

                    candidate = 1;
                } else if (secondShape[i] == 1) {
                    if (candidate == 1) {
                        return -1;
                    }

                    candidate = 2;
                } else {
                    return -1;
                }
            }
        }

        return candidate;
    }


    public static void broadcast(float[] input, int inputOffset, int[] inputShape,
                                 float[] output, int outputOffset, int[] outputShape) {
        if (outputShape.length < inputShape.length) {
            throw new IllegalArgumentException("Output shape must have at least the same rank as input shape");
        }

        inputShape = broadcastShape(inputShape, outputShape);

        if (isNotBroadcastCompatible(inputShape, outputShape)) {
            throw new IllegalArgumentException("Shapes are not broadcast compatible. Input shape: " +
                    Arrays.toString(inputShape) + ", output shape: " +
                    Arrays.toString(outputShape) + ".");
        }


        var batchRank = -1;

        for (int i = inputShape.length - 1; i >= 0; i--) {
            if (inputShape[i] != outputShape[i]) {
                batchRank = i;
                break;
            }
        }

        var inputStride = stride(inputShape);
        if (batchRank == -1) {
            System.arraycopy(input, inputOffset, output, outputOffset, inputStride);
            return;
        }

        var currentRank = 0;
        var outputStride = stride(outputShape);

        copyAndBroadcastDimension(input, inputOffset, inputStride / inputShape[currentRank],
                inputShape, output, outputOffset,
                outputStride / outputShape[currentRank],
                outputShape,
                currentRank, batchRank);
    }

    private static void copyAndBroadcastDimension(float[] input, int inputOffset, int inputStrideWidth,
                                                  int[] inputShape, float[] output, int outputOffset, int outputStrideWidth,
                                                  int[] outputShape, int currentRank, int batchRank) {
        assert currentRank <= batchRank;

        if (currentRank == batchRank) {
            System.arraycopy(input, inputOffset, output, outputOffset, inputStrideWidth);
            assert inputStrideWidth == outputStrideWidth;

            if (inputShape[currentRank] != outputShape[currentRank]) {
                assert inputShape[currentRank] == 1;

                duplicateDimension(output, outputOffset, outputStrideWidth, outputShape, currentRank);
            }
        } else {
            var inputDimension = inputShape[currentRank];
            var outputDimension = outputShape[currentRank];

            for (int i = 0; i < inputDimension; i++) {
                copyAndBroadcastDimension(input, inputOffset + i * inputStrideWidth,
                        inputStrideWidth / inputShape[currentRank + 1], inputShape,
                        output, outputOffset + i * outputStrideWidth,
                        outputStrideWidth / outputShape[currentRank + 1], outputShape,
                        currentRank + 1, batchRank);
            }

            if (inputDimension != outputDimension) {
                assert inputDimension == 1;
                duplicateDimension(output, outputOffset, outputStrideWidth, outputShape, currentRank);
            }
        }
    }

    private static void duplicateDimension(float[] output, int outputOffset,
                                           int outputStrideWidth, int[] outputShape, int currentRank) {
        var repeat = outputShape[currentRank];
        var outputIndex = outputOffset + outputStrideWidth;

        var batchSize = 1;
        int i = 1;

        while (i + batchSize <= repeat) {
            var batchWidth = batchSize * outputStrideWidth;

            System.arraycopy(output, outputOffset, output, outputIndex, batchWidth);

            outputIndex += batchWidth;
            i += batchSize;

            batchSize <<= 1;
        }

        if (i < repeat) {
            batchSize = repeat - i;
            System.arraycopy(output, outputOffset, output, outputIndex, batchSize * outputStrideWidth);
        }
    }

    public static int @NonNull [] broadcastShape(int[] inputShape, int[] outputShape) {
        if (outputShape.length > inputShape.length) {
            var newInputShape = new int[outputShape.length];

            for (int i = 0; i < outputShape.length - inputShape.length; i++) {
                newInputShape[i] = 1;
            }
            System.arraycopy(inputShape, 0, newInputShape, outputShape.length - inputShape.length,
                    inputShape.length);
            inputShape = newInputShape;
        }
        return inputShape;
    }


    public static void reduce(float[] input, int inputOffset, int[] inputShape, float[] output,
                              int outputOffset, int[] outputShape) {
        if (inputShape.length < outputShape.length) {
            throw new IllegalArgumentException("Input shape must have at least the same rank as output shape");
        }

        outputShape = broadcastShape(outputShape, inputShape);

        if (isNotBroadcastCompatible(outputShape, inputShape)) {
            throw new IllegalArgumentException("Shapes are not reduce compatible. Input shape: " +
                    Arrays.toString(inputShape) + ", output shape: " +
                    Arrays.toString(outputShape) + ".");
        }


        var batchRank = -1;

        for (int i = inputShape.length - 1; i >= 0; i--) {
            if (inputShape[i] != outputShape[i]) {
                batchRank = i;
                break;
            }
        }

        var inputStride = stride(inputShape);
        if (batchRank == -1) {
            System.arraycopy(input, inputOffset, output, outputOffset, inputStride);
            return;
        }

        var currentRank = 0;
        var outputStride = stride(outputShape);

        copyAndReduceDimension(input, inputOffset, inputStride / inputShape[currentRank],
                inputShape, output, outputOffset,
                outputStride / outputShape[currentRank],
                outputShape,
                currentRank, batchRank);

    }

    private static void copyAndReduceDimension(float[] input, int inputOffset, int inputStrideWidth,
                                               int[] inputShape, float[] output, int outputOffset, int outputStrideWidth,
                                               int[] outputShape, int currentRank, int batchRank) {
        assert currentRank <= batchRank;

        if (currentRank == batchRank) {
            System.arraycopy(input, inputOffset, output, outputOffset, inputStrideWidth);
            assert inputStrideWidth == outputStrideWidth;

            if (inputShape[currentRank] != outputShape[currentRank]) {
                assert outputShape[currentRank] == 1;

                reduceDimension(input, inputOffset, inputShape, inputStrideWidth,
                        output, outputOffset, outputStrideWidth, outputShape, currentRank);
            }
        } else {
            var outputDimension = outputShape[currentRank];

            for (int i = 0; i < outputDimension; i++) {
                copyAndReduceDimension(input, inputOffset + i * inputStrideWidth,
                        inputStrideWidth / inputShape[currentRank + 1], inputShape,
                        output, outputOffset + i * outputStrideWidth,
                        outputStrideWidth / outputShape[currentRank + 1], outputShape,
                        currentRank + 1, batchRank);
            }
        }
    }

    private static void reduceDimension(float[] input, int inputOffset, int[] inputShape,
                                        int inputStrideWidth, float[] output, int outputOffset,
                                        int outputStrideWidth, int[] outputShape, int currentRank) {
        assert outputShape[currentRank] == 1;
        assert inputStrideWidth == outputStrideWidth;

        var repeat = inputShape[currentRank];
        for (int i = 0; i < repeat; i++) {
            var inputIndex = inputOffset + i * inputStrideWidth;
            VectorOperations.addVectorToVector(input, inputIndex,
                    output, outputOffset, input, inputIndex, inputStrideWidth);
        }

    }
}
