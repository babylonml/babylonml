package com.babylonml.backend.cpu;


import org.jspecify.annotations.NonNull;

import java.util.Arrays;

public abstract class TensorOperations {
    public static int stride(int @NonNull [] shape) {
        int stride = 1;
        for (int dim : shape) {
            stride *= dim;
        }
        return stride;
    }

    public static int stride(@NonNull Object data) {
        int stride = 1;

        if (data instanceof Object[] objects) {
            stride *= objects.length;
            stride *= stride(objects[0]);
        } else if (data instanceof float[] floats) {
            stride *= floats.length;
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + data.getClass());
        }

        return stride;
    }

    public static int @NonNull [] calculateMaxShape(int @NonNull [] leftShape, int @NonNull [] rightShape) {
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

    private static boolean isNotBroadcastCompatible(int @NonNull [] firstShape, int @NonNull [] secondShape) {
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
    public static int broadcastCandidate(int @NonNull [] firstShape, int @NonNull [] secondShape) {
        int candidate = 0;

        if (firstShape.length < secondShape.length) {
            candidate = 1;
        } else if (firstShape.length > secondShape.length) {
            candidate = 2;
        }

        if (candidate == 1) {
            firstShape = broadcastShape(firstShape, secondShape);
        } else if (candidate == 2) {
            secondShape = broadcastShape(secondShape, firstShape);
        }

        for (int i = 0; i < firstShape.length; i++) {
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


    public static void broadcast(float @NonNull [] input, int inputOffset, int @NonNull [] inputShape,
                                 float @NonNull [] output, int outputOffset, int @NonNull [] outputShape) {
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

    private static void copyAndBroadcastDimension(float @NonNull [] input, int inputOffset, int inputStrideWidth,
                                                  int @NonNull [] inputShape, float @NonNull [] output,
                                                  int outputOffset, int outputStrideWidth,
                                                  int @NonNull [] outputShape, int currentRank, int batchRank) {
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

    private static void duplicateDimension(float @NonNull [] output, int outputOffset,
                                           int outputStrideWidth, int @NonNull [] outputShape, int currentRank) {
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

    public static int @NonNull [] broadcastShape(int @NonNull [] shapeToBroadcast, int @NonNull [] templateShape) {
        if (templateShape.length > shapeToBroadcast.length) {
            var newInputShape = new int[templateShape.length];

            for (int i = 0; i < templateShape.length - shapeToBroadcast.length; i++) {
                newInputShape[i] = 1;
            }

            System.arraycopy(shapeToBroadcast, 0, newInputShape, templateShape.length - shapeToBroadcast.length,
                    shapeToBroadcast.length);

            shapeToBroadcast = newInputShape;
        }

        return shapeToBroadcast;
    }


    public static void reduce(float @NonNull [] input, int inputOffset, int @NonNull [] inputShape,
                              float @NonNull [] output, int outputOffset, int @NonNull [] outputShape) {
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

        for (int i = outputShape.length - 1; i >= 0; i--) {
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

    private static void copyAndReduceDimension(float @NonNull [] input, int inputOffset, int inputStrideWidth,
                                               int @NonNull [] inputShape, float @NonNull [] output,
                                               int outputOffset, int outputStrideWidth,
                                               int @NonNull [] outputShape, int currentRank, int batchRank) {
        assert currentRank <= batchRank;

        if (currentRank == batchRank) {
            System.arraycopy(input, inputOffset, output, outputOffset, inputStrideWidth);
            assert inputStrideWidth == outputStrideWidth;

            if (inputShape[currentRank] != outputShape[currentRank]) {
                assert outputShape[currentRank] == 1;

                var repeat = inputShape[currentRank];
                for (int i = 1; i < repeat; i++) {
                    var inputIndex = inputOffset + i * inputStrideWidth;
                    VectorOperations.addVectorToVector(input, inputIndex,
                            output, outputOffset, output, outputOffset, outputStrideWidth);
                }
            }
        } else {
            var outputDimension = outputShape[currentRank];
            var inputDimension = inputShape[currentRank];

            if (inputDimension == outputDimension) {
                for (int i = 0; i < outputDimension; i++) {
                    copyAndReduceDimension(input, inputOffset + i * inputStrideWidth,
                            inputStrideWidth / inputShape[currentRank + 1], inputShape,
                            output, outputOffset + i * outputStrideWidth,
                            outputStrideWidth / outputShape[currentRank + 1], outputShape,
                            currentRank + 1, batchRank);
                }
            } else {
                assert outputDimension == 1;

                copyAndReduceDimension(input, inputOffset,
                        inputStrideWidth / inputShape[currentRank + 1], inputShape,
                        output, outputOffset,
                        outputStrideWidth / outputShape[currentRank + 1], outputShape,
                        currentRank + 1, batchRank);
                for (int i = 1; i < inputDimension; i++) {
                    reduceDimension(input, inputOffset + i * inputStrideWidth,
                            inputStrideWidth / inputShape[currentRank + 1], inputShape,
                            output, outputOffset,
                            outputStrideWidth / outputShape[currentRank + 1], outputShape,
                            currentRank + 1, batchRank);
                }
            }
        }
    }

    private static void reduceDimension(float @NonNull [] input, int inputOffset, int inputStrideWidth,
                                        int @NonNull [] inputShape, float @NonNull [] output,
                                        int outputOffset, int outputStrideWidth,
                                        int @NonNull [] outputShape, int currentRank, int batchRank) {
        var outputDimension = outputShape[currentRank];
        var inputDimension = inputShape[currentRank];

        if (currentRank == batchRank) {
            if (inputDimension != outputDimension) {
                assert outputShape[currentRank] == 1;

                var repeat = inputShape[currentRank];
                for (int i = 0; i < repeat; i++) {
                    var inputIndex = inputOffset + i * inputStrideWidth;
                    VectorOperations.addVectorToVector(input, inputIndex,
                            output, outputOffset, output, outputOffset, outputStrideWidth);
                }
            }
        } else {
            if (inputDimension == outputDimension) {
                for (int i = 0; i < inputDimension; i++) {
                    reduceDimension(input, inputOffset + i * inputStrideWidth,
                            inputStrideWidth / inputShape[currentRank + 1], inputShape,
                            output, outputOffset + i * outputStrideWidth,
                            outputStrideWidth / outputShape[currentRank + 1], outputShape,
                            currentRank + 1, batchRank);
                }
            } else {
                assert outputDimension == 1;
                for (int i = 0; i < inputDimension; i++) {
                    reduceDimension(input, inputOffset + i * inputStrideWidth,
                            inputStrideWidth / inputShape[currentRank + 1], inputShape,
                            output, outputOffset,
                            outputStrideWidth / outputShape[currentRank + 1], outputShape,
                            currentRank + 1, batchRank);
                }
            }
        }
    }

    public static void bmm(float @NonNull [] first, int firstOffset, int @NonNull [] firstShape,
                           float @NonNull [] second, int secondOffset,
                           int @NonNull [] secondShape, float @NonNull [] result, int resultOffset,
                           int @NonNull [] resultShape) {
        validateBMMShapesValidity(firstShape, secondShape);

        var diff = firstShape.length - 2;
        for (int i = 0; i < diff; i++) {
            if (firstShape[i] != resultShape[i]) {
                throw new IllegalArgumentException("Input and result shapes must have the" +
                        " same dimensions for the first " + diff + " dimensions");
            }
        }

        if (resultShape.length != firstShape.length) {
            throw new IllegalArgumentException("Result shape must have the same rank as first and second shapes");
        }
        if (resultShape[diff] != firstShape[diff]) {
            throw new IllegalArgumentException("Result shape must have the same second to last dimension as the first shape");
        }
        if (resultShape[diff + 1] != secondShape[diff + 1]) {
            throw new IllegalArgumentException("Result shape must have the same last dimension as the second shape");
        }


        var firstWidth = 1;
        for (int i = diff; i < firstShape.length; i++) {
            firstWidth *= firstShape[i];
        }

        var secondWidth = 1;
        for (int i = diff; i < secondShape.length; i++) {
            secondWidth *= secondShape[i];
        }

        var resultWidth = 1;
        for (int i = diff; i < resultShape.length; i++) {
            resultWidth *= resultShape[i];
        }

        var matrices = 1;
        for (int i = 0; i < diff; i++) {
            matrices *= firstShape[i];
        }

        for (int i = 0; i < matrices; i++) {
            var firstIndex = firstOffset + i * firstWidth;
            var secondIndex = secondOffset + i * secondWidth;
            var resultIndex = resultOffset + i * resultWidth;

            MatrixOperations.matrixToMatrixMultiplication(first, firstIndex, firstShape[diff], firstShape[diff + 1],
                    second, secondIndex, secondShape[diff], secondShape[diff + 1],
                    result, resultIndex);
        }
    }

    private static void validateBMMShapesValidity(int @NonNull [] firstShape, int @NonNull [] secondShape) {
        if (firstShape.length < 2) {
            throw new IllegalArgumentException("First and second shapes must have at least 2 dimensions");
        }
        if (firstShape.length != secondShape.length) {
            throw new IllegalArgumentException("First and second shapes must have the same rank. First shape: "
                    + Arrays.toString(firstShape) +
                    ", second shape: " + Arrays.toString(secondShape) + ".");
        }

        var diff = firstShape.length - 2;
        for (int i = 0; i < diff; i++) {
            if (firstShape[i] != secondShape[i]) {
                throw new IllegalArgumentException("First and second shapes must have the" +
                        " same dimensions for the first " + diff + " dimensions. " +
                        "First shape: " + Arrays.toString(firstShape) +
                        ", second shape: " + Arrays.toString(secondShape) + ".");
            }

        }
        if (firstShape[diff + 1] != secondShape[diff]) {
            throw new IllegalArgumentException("Second to last dimension of first shape must be equal to" +
                    " the last dimension of the second shape. First shape: " + Arrays.toString(firstShape) +
                    ", second shape: " + Arrays.toString(secondShape) + ".");
        }
    }

    public static int[] calculateBMMShape(int[] firstShape, int[] secondShape) {
        validateBMMShapesValidity(firstShape, secondShape);

        var resultShape = new int[firstShape.length];
        System.arraycopy(firstShape, 0, resultShape, 0, firstShape.length - 2);

        resultShape[firstShape.length - 2] = firstShape[firstShape.length - 2];
        resultShape[firstShape.length - 1] = secondShape[secondShape.length - 1];

        return resultShape;
    }

    public static void bmt(float @NonNull [] input, int inputOffset, int @NonNull [] inputShape,
                           float @NonNull [] result, int resultOffset, int @NonNull [] resultShape) {
        if (inputShape.length < 2) {
            throw new IllegalArgumentException("Input shape must have at least 2 dimensions");
        }

        if (resultShape.length != inputShape.length) {
            throw new IllegalArgumentException("Result shape must have the same rank as input shape.");
        }

        var diff = inputShape.length - 2;
        for (int i = 0; i < diff; i++) {
            if (inputShape[i] != resultShape[i]) {
                throw new IllegalArgumentException("Input and result shapes must have the" +
                        " same dimensions for the first " + diff + " dimensions");
            }
        }

        if (resultShape[diff + 1] != inputShape[diff]) {
            throw new IllegalArgumentException("Second to last dimension of input shape must be equal to" +
                    " the last dimension of the result shape. Input shape: " + Arrays.toString(inputShape) +
                    ", result shape: " + Arrays.toString(resultShape) + ".");
        }
        if (resultShape[diff] != inputShape[diff + 1]) {
            throw new IllegalArgumentException("Result shape must have the same second to last dimension as " +
                    "the last dimension of input shape. " +
                    "Input shape: " + Arrays.toString(inputShape) +
                    ", result shape: " + Arrays.toString(resultShape) + ".");
        }

        var inputWidth = 1;
        for (int i = diff; i < inputShape.length; i++) {
            inputWidth *= inputShape[i];
        }


        var resultWidth = 1;
        for (int i = diff; i < resultShape.length; i++) {
            resultWidth *= resultShape[i];
        }

        var matrices = 1;
        for (int i = 0; i < diff; i++) {
            matrices *= inputShape[i];
        }

        for (int i = 0; i < matrices; i++) {
            var inputIndex = inputOffset + i * inputWidth;
            var resultIndex = resultOffset + i * resultWidth;

            MatrixOperations.transposeMatrix(input, inputIndex, inputShape[diff], inputShape[diff + 1],
                    result, resultIndex);
        }
    }

    public static int[] calculateBMTShape(int @NonNull [] shape) {

        if (shape.length < 2) {
            throw new IllegalArgumentException("Shape must have at least 2 dimensions. Shape : " +
                    Arrays.toString(shape));
        }

        var resultShape = new int[shape.length];

        System.arraycopy(shape, 0, resultShape, 0, shape.length - 2);
        resultShape[shape.length - 2] = shape[shape.length - 1];
        resultShape[shape.length - 1] = shape[shape.length - 2];

        return resultShape;
    }
}
