package com.babylonml.backend.cpu;


import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;

public abstract class TensorOperations {
    public static int stride(int @NonNull [] shape) {
        int stride = 1;
        for (int dim : shape) {
            stride *= dim;
        }
        return stride;
    }

    public static int stride(@NonNull IntImmutableList shape) {
        int stride = 1;
        for (int i = 0; i < shape.size(); i++) {
            stride *= shape.getInt(i);
        }
        return stride;
    }

    public static @NonNull IntImmutableList calculateMaxShape(@NonNull IntImmutableList leftShape,
                                                              @NonNull IntImmutableList rightShape) {
        if (leftShape.size() == rightShape.size()) {
            var maxShape = new int[leftShape.size()];

            for (int i = 0; i < leftShape.size(); i++) {
                maxShape[i] = Math.max(leftShape.getInt(i), rightShape.getInt(i));
            }

            return new IntImmutableList(maxShape);
        }

        if (leftShape.size() < rightShape.size()) {
            var left = extendShapeTill(leftShape, rightShape.size());
            return calculateMaxShape(left, rightShape);
        }

        var right = extendShapeTill(rightShape, leftShape.size());
        return calculateMaxShape(leftShape, right);
    }

    private static boolean isNotBroadcastCompatible(@NonNull IntImmutableList firstShape,
                                                    @NonNull IntImmutableList secondShape) {
        for (int i = 0; i < secondShape.size(); i++) {
            if (firstShape.getInt(i) != secondShape.getInt(i) &&
                    firstShape.getInt(i) != 1 && secondShape.getInt(i) != 1) {
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
    public static int broadcastCandidate(@NonNull IntImmutableList firstShape, @NonNull IntImmutableList secondShape) {
        int candidate = 0;

        if (firstShape.size() < secondShape.size()) {
            candidate = 1;
        } else if (firstShape.size() > secondShape.size()) {
            candidate = 2;
        }

        if (candidate == 1) {
            firstShape = extendShapeTill(firstShape, secondShape.size());
        } else if (candidate == 2) {
            secondShape = extendShapeTill(secondShape, firstShape.size());
        }

        for (int i = 0; i < firstShape.size(); i++) {
            if (firstShape.getInt(i) != secondShape.getInt(i)) {
                if (firstShape.getInt(i) == 1) {
                    if (candidate == 2) {
                        return -1;
                    }

                    candidate = 1;
                } else if (secondShape.getInt(i) == 1) {
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


    public static void broadcast(float @NonNull [] input, int inputOffset, @NonNull IntImmutableList inputShape,
                                 float @NonNull [] output, int outputOffset, @NonNull IntImmutableList outputShape,
                                 int broadcastTillRank) {
        if (broadcastTillRank < 0) {
            broadcastTillRank = inputShape.size();
        }

        if (broadcastTillRank > inputShape.size()) {
            throw new IllegalArgumentException("Broadcast till rank must be less than or equal" +
                    " to the rank of the input shape. Requested broadcast till rank: " +
                    broadcastTillRank + ", input shape rank: " +
                    inputShape.size() + ".");
        }

        if (outputShape.size() < inputShape.size()) {
            throw new IllegalArgumentException("Output shape must have at least the same rank as input shape");
        }

        var prevSize = inputShape.size();
        inputShape = extendShapeTill(inputShape, outputShape.size());

        //adjust broadcastTillRank to the new shape
        broadcastTillRank += inputShape.size() - prevSize;

        if (outputShape.size() > broadcastTillRank) {
            var modifiedShape = new int[outputShape.size()];
            outputShape.getElements(0, modifiedShape, 0, broadcastTillRank);
            inputShape.getElements(broadcastTillRank, modifiedShape, broadcastTillRank,
                    outputShape.size() - broadcastTillRank);

            outputShape = IntImmutableList.of(modifiedShape);
        }

        if (isNotBroadcastCompatible(inputShape, outputShape)) {
            throw new IllegalArgumentException("Shapes are not broadcast compatible. Input shape: " +
                    inputShape + ", output shape: " +
                    outputShape + ". Broadcast till rank: " + broadcastTillRank + ".");
        }

        var batchRank = -1;
        for (int i = broadcastTillRank - 1; i >= 0; i--) {
            if (inputShape.getInt(i) != outputShape.getInt(i)) {
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

        copyAndBroadcastDimension(input, inputOffset, inputStride / inputShape.getInt(currentRank),
                inputShape, output, outputOffset,
                outputStride / outputShape.getInt(currentRank),
                outputShape,
                currentRank, batchRank);
    }

    private static void copyAndBroadcastDimension(float @NonNull [] input, int inputOffset, int inputStrideWidth,
                                                  @NonNull IntImmutableList inputShape, float @NonNull [] output,
                                                  int outputOffset, int outputStrideWidth,
                                                  @NonNull IntImmutableList outputShape,
                                                  int currentRank, int batchRank) {
        assert currentRank <= batchRank;

        if (currentRank == batchRank) {
            System.arraycopy(input, inputOffset, output, outputOffset, inputStrideWidth);
            assert inputStrideWidth == outputStrideWidth;

            if (inputShape.getInt(currentRank) != outputShape.getInt(currentRank)) {
                assert inputShape.getInt(currentRank) == 1;

                duplicateDimension(output, outputOffset, outputStrideWidth, outputShape, currentRank);
            }
        } else {
            var inputDimension = inputShape.getInt(currentRank);
            var outputDimension = outputShape.getInt(currentRank);

            for (int i = 0; i < inputDimension; i++) {
                copyAndBroadcastDimension(input, inputOffset + i * inputStrideWidth,
                        inputStrideWidth / inputShape.getInt(currentRank + 1), inputShape,
                        output, outputOffset + i * outputStrideWidth,
                        outputStrideWidth / outputShape.getInt(currentRank + 1), outputShape,
                        currentRank + 1, batchRank);
            }

            if (inputDimension != outputDimension) {
                assert inputDimension == 1;
                duplicateDimension(output, outputOffset, outputStrideWidth, outputShape, currentRank);
            }
        }
    }

    private static void duplicateDimension(float @NonNull [] output, int outputOffset,
                                           int outputStrideWidth, @NonNull IntImmutableList outputShape, int currentRank) {
        var repeat = outputShape.getInt(currentRank);
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

    public static @NonNull IntImmutableList extendShapeTill(@NonNull IntImmutableList shapeToBroadcast, int size) {
        if (size > shapeToBroadcast.size()) {
            var newInputShape = new int[size];

            for (int i = 0; i < size - shapeToBroadcast.size(); i++) {
                newInputShape[i] = 1;
            }

            shapeToBroadcast.getElements(0, newInputShape, size - shapeToBroadcast.size(),
                    shapeToBroadcast.size());

            return new IntImmutableList(newInputShape);
        } else if (size < shapeToBroadcast.size()) {
            throw new IllegalArgumentException("Size must be greater than or equal to the size of the shape to broadcast");
        }

        return shapeToBroadcast;
    }


    public static void reduce(float @NonNull [] input, int inputOffset, @NonNull IntImmutableList inputShape,
                              float @NonNull [] output, int outputOffset, @NonNull IntImmutableList outputShape) {
        if (inputShape.size() < outputShape.size()) {
            throw new IllegalArgumentException("Input shape must have at least the same rank as output shape");
        }

        outputShape = extendShapeTill(outputShape, inputShape.size());

        if (isNotBroadcastCompatible(outputShape, inputShape)) {
            throw new IllegalArgumentException("Shapes are not reduce compatible. Input shape: " +
                    inputShape + ", output shape: " +
                    outputShape + ".");
        }


        var batchRank = -1;

        for (int i = outputShape.size() - 1; i >= 0; i--) {
            if (inputShape.getInt(i) != outputShape.getInt(i)) {
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

        copyAndReduceDimension(input, inputOffset, inputStride / inputShape.getInt(currentRank),
                inputShape, output, outputOffset,
                outputStride / outputShape.getInt(currentRank),
                outputShape,
                currentRank, batchRank);

    }

    private static void copyAndReduceDimension(float @NonNull [] input, int inputOffset, int inputStrideWidth,
                                               @NonNull IntImmutableList inputShape, float @NonNull [] output,
                                               int outputOffset, int outputStrideWidth,
                                               @NonNull IntImmutableList outputShape, int currentRank, int batchRank) {
        assert currentRank <= batchRank;

        if (currentRank == batchRank) {
            System.arraycopy(input, inputOffset, output, outputOffset, inputStrideWidth);
            assert inputStrideWidth == outputStrideWidth;

            if (inputShape.getInt(currentRank) != outputShape.getInt(currentRank)) {
                assert outputShape.getInt(currentRank) == 1;

                var repeat = inputShape.getInt(currentRank);
                for (int i = 1; i < repeat; i++) {
                    var inputIndex = inputOffset + i * inputStrideWidth;
                    VectorOperations.addVectorToVector(input, inputIndex,
                            output, outputOffset, output, outputOffset, outputStrideWidth);
                }
            }
        } else {
            var outputDimension = outputShape.getInt(currentRank);
            var inputDimension = inputShape.getInt(currentRank);

            if (inputDimension == outputDimension) {
                for (int i = 0; i < outputDimension; i++) {
                    copyAndReduceDimension(input, inputOffset + i * inputStrideWidth,
                            inputStrideWidth / inputShape.getInt(currentRank + 1), inputShape,
                            output, outputOffset + i * outputStrideWidth,
                            outputStrideWidth / outputShape.getInt(currentRank + 1), outputShape,
                            currentRank + 1, batchRank);
                }
            } else {
                assert outputDimension == 1;

                copyAndReduceDimension(input, inputOffset,
                        inputStrideWidth / inputShape.getInt(currentRank + 1), inputShape,
                        output, outputOffset,
                        outputStrideWidth / outputShape.getInt(currentRank + 1), outputShape,
                        currentRank + 1, batchRank);
                for (int i = 1; i < inputDimension; i++) {
                    reduceDimension(input, inputOffset + i * inputStrideWidth,
                            inputStrideWidth / inputShape.getInt(currentRank + 1), inputShape,
                            output, outputOffset,
                            outputStrideWidth / outputShape.getInt(currentRank + 1), outputShape,
                            currentRank + 1, batchRank);
                }
            }
        }
    }

    private static void reduceDimension(float @NonNull [] input, int inputOffset, int inputStrideWidth,
                                        @NonNull IntImmutableList inputShape, float @NonNull [] output,
                                        int outputOffset, int outputStrideWidth,
                                        @NonNull IntImmutableList outputShape, int currentRank, int batchRank) {
        var outputDimension = outputShape.getInt(currentRank);
        var inputDimension = inputShape.getInt(currentRank);

        if (currentRank == batchRank) {
            if (inputDimension != outputDimension) {
                assert outputShape.getInt(currentRank) == 1;

                var repeat = inputShape.getInt(currentRank);
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
                            inputStrideWidth / inputShape.getInt(currentRank + 1), inputShape,
                            output, outputOffset + i * outputStrideWidth,
                            outputStrideWidth / outputShape.getInt(currentRank + 1), outputShape,
                            currentRank + 1, batchRank);
                }
            } else {
                assert outputDimension == 1;
                for (int i = 0; i < inputDimension; i++) {
                    reduceDimension(input, inputOffset + i * inputStrideWidth,
                            inputStrideWidth / inputShape.getInt(currentRank + 1), inputShape,
                            output, outputOffset,
                            outputStrideWidth / outputShape.getInt(currentRank + 1), outputShape,
                            currentRank + 1, batchRank);
                }
            }
        }
    }

    public static void bmm(float @NonNull [] first, int firstOffset, @NonNull IntImmutableList firstShape,
                           float @NonNull [] second, int secondOffset,
                           @NonNull IntImmutableList secondShape, float @NonNull [] result, int resultOffset,
                           @NonNull IntImmutableList resultShape) {
        validateBMMShapesValidity(firstShape, secondShape);

        var diff = firstShape.size() - 2;
        for (int i = 0; i < diff; i++) {
            if (firstShape.getInt(i) != resultShape.getInt(i)) {
                throw new IllegalArgumentException("Input and result shapes must have the" +
                        " same dimensions for the first " + diff + " dimensions");
            }
        }

        if (resultShape.size() != firstShape.size()) {
            throw new IllegalArgumentException("Result shape must have the same rank as first and second shapes");
        }
        if (resultShape.getInt(diff) != firstShape.getInt(diff)) {
            throw new IllegalArgumentException("Result shape must have the same second to last dimension as the first shape");
        }
        if (resultShape.getInt(diff + 1) != secondShape.getInt(diff + 1)) {
            throw new IllegalArgumentException("Result shape must have the same last dimension as the second shape");
        }


        var firstWidth = 1;
        for (int i = diff; i < firstShape.size(); i++) {
            firstWidth *= firstShape.getInt(i);
        }

        var secondWidth = 1;
        for (int i = diff; i < secondShape.size(); i++) {
            secondWidth *= secondShape.getInt(i);
        }

        var resultWidth = 1;
        for (int i = diff; i < resultShape.size(); i++) {
            resultWidth *= resultShape.getInt(i);
        }

        var matrices = 1;
        for (int i = 0; i < diff; i++) {
            matrices *= firstShape.getInt(i);
        }

        for (int i = 0; i < matrices; i++) {
            var firstIndex = firstOffset + i * firstWidth;
            var secondIndex = secondOffset + i * secondWidth;
            var resultIndex = resultOffset + i * resultWidth;

            MatrixOperations.matrixToMatrixMultiplication(first, firstIndex, firstShape.getInt(diff),
                    firstShape.getInt(diff + 1), second, secondIndex, secondShape.getInt(diff),
                    secondShape.getInt(diff + 1), result, resultIndex);
        }
    }

    private static void validateBMMShapesValidity(@NonNull IntImmutableList firstShape, @NonNull IntImmutableList secondShape) {
        if (firstShape.size() < 2) {
            throw new IllegalArgumentException("First and second shapes must have at least 2 dimensions");
        }
        if (firstShape.size() != secondShape.size()) {
            throw new IllegalArgumentException("First and second shapes must have the same rank. First shape: "
                    + firstShape +
                    ", second shape: " + secondShape + ".");
        }

        var diff = firstShape.size() - 2;
        for (int i = 0; i < diff; i++) {
            if (firstShape.getInt(i) != secondShape.getInt(i)) {
                throw new IllegalArgumentException("First and second shapes must have the" +
                        " same dimensions for the first " + diff + " dimensions. " +
                        "First shape: " + firstShape +
                        ", second shape: " + secondShape + ".");
            }

        }
        if (firstShape.getInt(diff + 1) != secondShape.getInt(diff)) {
            throw new IllegalArgumentException("Second to last dimension of first shape must be equal to" +
                    " the last dimension of the second shape. First shape: " + firstShape +
                    ", second shape: " + secondShape + ".");
        }
    }

    public static IntImmutableList calculateBMMShape(IntImmutableList firstShape, IntImmutableList secondShape) {
        validateBMMShapesValidity(firstShape, secondShape);

        var resultShape = new int[firstShape.size()];
        firstShape.getElements(0, resultShape, 0, firstShape.size() - 2);

        resultShape[firstShape.size() - 2] = firstShape.getInt(firstShape.size() - 2);
        resultShape[firstShape.size() - 1] = secondShape.getInt(secondShape.size() - 1);

        return IntImmutableList.of(resultShape);
    }

    public static void bmt(float @NonNull [] input, int inputOffset, @NonNull IntImmutableList inputShape,
                           float @NonNull [] result, int resultOffset, @NonNull IntImmutableList resultShape) {
        if (inputShape.size() < 2) {
            throw new IllegalArgumentException("Input shape must have at least 2 dimensions");
        }

        if (resultShape.size() != inputShape.size()) {
            throw new IllegalArgumentException("Result shape must have the same rank as input shape.");
        }

        var diff = inputShape.size() - 2;
        for (int i = 0; i < diff; i++) {
            if (inputShape.getInt(i) != resultShape.getInt(i)) {
                throw new IllegalArgumentException("Input and result shapes must have the" +
                        " same dimensions for the first " + diff + " dimensions");
            }
        }

        if (resultShape.getInt(diff + 1) != inputShape.getInt(diff)) {
            throw new IllegalArgumentException("Second to last dimension of input shape must be equal to" +
                    " the last dimension of the result shape. Input shape: " + inputShape +
                    ", result shape: " + resultShape + ".");
        }
        if (resultShape.getInt(diff) != inputShape.getInt(diff + 1)) {
            throw new IllegalArgumentException("Result shape must have the same second to last dimension as " +
                    "the last dimension of input shape. " +
                    "Input shape: " + inputShape +
                    ", result shape: " + resultShape + ".");
        }

        var inputWidth = 1;
        for (int i = diff; i < inputShape.size(); i++) {
            inputWidth *= inputShape.getInt(i);
        }


        var resultWidth = 1;
        for (int i = diff; i < resultShape.size(); i++) {
            resultWidth *= resultShape.getInt(i);
        }

        var matrices = 1;
        for (int i = 0; i < diff; i++) {
            matrices *= inputShape.getInt(i);
        }

        for (int i = 0; i < matrices; i++) {
            var inputIndex = inputOffset + i * inputWidth;
            var resultIndex = resultOffset + i * resultWidth;

            MatrixOperations.transposeMatrix(input, inputIndex, inputShape.getInt(diff), inputShape.getInt(diff + 1),
                    result, resultIndex);
        }
    }

    public static IntImmutableList calculateBMTShape(@NonNull IntImmutableList shape) {

        if (shape.size() < 2) {
            throw new IllegalArgumentException("Shape must have at least 2 dimensions. Shape : " +
                    shape);
        }

        var resultShape = new int[shape.size()];

        shape.getElements(0, resultShape, 0, shape.size() - 2);
        resultShape[shape.size() - 2] = shape.getInt(shape.size() - 1);
        resultShape[shape.size() - 1] = shape.getInt(shape.size() - 2);

        return IntImmutableList.of(resultShape);
    }
}
