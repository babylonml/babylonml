package com.babylonml.backend.tornadovm;

import com.babylonml.backend.common.CommonTensorOperations;
import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

public class TvmTensorOperations {
    public static void addBroadcastTask(TaskGraph taskGraph, String prefix, @NonNull FloatArray input,
                                        int inputOffset, @NonNull IntImmutableList inputShape,
                                        @NonNull FloatArray output, int outputOffset, @NonNull IntImmutableList outputShape) {


        if (outputShape.size() < inputShape.size()) {
            throw new IllegalArgumentException("Output shape must have at least the same rank as input shape");
        }

        inputShape = CommonTensorOperations.extendShapeTill(inputShape, outputShape.size());

        //adjust broadcastTillRank to the new shape

        if (CommonTensorOperations.isNotBroadcastCompatible(inputShape, outputShape)) {
            throw new IllegalArgumentException("Shapes are not broadcast compatible. Input shape: " +
                    inputShape + ", output shape: " +
                    outputShape + ".");
        }

        var batchRank = -1;
        for (int i = inputShape.size() - 1; i >= 0; i--) {
            if (inputShape.getInt(i) != outputShape.getInt(i)) {
                batchRank = i;
                break;
            }
        }

        if (batchRank == -1) {
            var inputStride = CommonTensorOperations.stride(inputShape);
            TvmVectorOperations.addCopyVectorTask(taskGraph,
                    TvmCommons.generateName(prefix + "-copyVectorForBroadcast"), input, inputOffset,
                    output, outputOffset, inputStride);
            return;
        }

        var outputStride = CommonTensorOperations.stride(outputShape);
        var inputShapeArray = IntArray.fromArray(inputShape.toIntArray());
        var outputShapeArray = IntArray.fromArray(outputShape.toIntArray());

        taskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, inputShapeArray, outputShapeArray);
        taskGraph.task(TvmCommons.generateName(prefix + "-broadcastKernel"),
                TvmTensorOperations::broadcastKernel,
                input, inputShapeArray, inputOffset, output, outputOffset, outputStride, outputShapeArray);
    }

    static void broadcastKernel(@NonNull FloatArray input, @NonNull IntArray inputShape, int inputOffset,
                                @NonNull FloatArray output, int outputOffset, int outputStrideWidth,
                                @NonNull IntArray outputShape) {
        final int outputShapeSize = outputShape.getSize();
        for (int outputIndex = 0; outputIndex < outputStrideWidth; outputIndex++) {
            int currentOutputStrideWidth = 1;
            int currentInputStrideWidth = 1;

            int inputIndex = 0;
            for (int currentRank = outputShapeSize - 1; currentRank >= 0; currentRank--) {
                var internalIndexOutputIndex = outputIndex / currentOutputStrideWidth;

                int outputDimension = outputShape.get(currentRank);
                int inputDimension = inputShape.get(currentRank);

                var externalIndexOutputIndex = internalIndexOutputIndex % outputDimension;
                //gpu does not like ifs so we calculate the index and then multiply by 0 or 1
                var internalIndexInputIndex = externalIndexOutputIndex * (inputDimension / outputDimension);

                inputIndex += internalIndexInputIndex * currentInputStrideWidth;

                currentInputStrideWidth *= inputDimension;
                currentOutputStrideWidth *= outputDimension;
            }

            output.set(outputOffset + outputIndex, input.get(inputOffset + inputIndex));
        }
    }

    static void ropeKernel(@NonNull FloatArray input, @NonNull IntArray inputShape, final int inputOffset,
                           @NonNull FloatArray cosArray, final int cosOffset, @NonNull FloatArray sinArray,
                           final int sinOffset, @NonNull FloatArray startPosition, int startPositionOffset,
                           @NonNull FloatArray result, final int resultOffset, int maxSequenceSize) {
        //cos/sin array is tensor with shape [positions, headDimension]
        //input is tensor with shape [batchSize, sequenceSize, numHeads, headDimension]
        //we are interested in head dimension only, so we distill all dimension into the list of
        //[maxSequenceSize, headDimension] arrays and apply RoPE to each of them

        final int batchSize = inputShape.get(0);
        final int sequenceSize = inputShape.get(1);

        final int numberOfHeads = inputShape.get(2);
        final int headDimension = inputShape.get(3);

        final int sequenceStepSize = numberOfHeads * headDimension;
        final int batchStepSize = sequenceStepSize * sequenceSize;

        final int halfHeadDim = headDimension / 2;

        final int batchSequenceIterations = batchSize * sequenceSize;

        for (@Parallel int batchSequenceIteration = 0; batchSequenceIteration < batchSequenceIterations; batchSequenceIteration++) {
            final int batchIndex = batchSequenceIteration / sequenceSize;
            final int sequenceIndex = batchSequenceIteration % sequenceSize;
            final int batchOffset = batchIndex * batchStepSize;
            final int position = (int) startPosition.get(startPositionOffset) + sequenceIndex;
            final int currentCosOffset = headDimension * position + cosOffset;
            final int currentSinOffset = headDimension * position + sinOffset;

            final int sequenceOffset = batchOffset + sequenceIndex * sequenceStepSize;
            for (@Parallel int h = 0; h < numberOfHeads; h++) {
                final int commonOffset = sequenceOffset + h * headDimension;
                final int inputTensorOffset = inputOffset + commonOffset;
                final int outputTensorOffset = resultOffset + commonOffset;

                for (@Parallel int i = 0; i < halfHeadDim; i++) {
                    float cosValue = cosArray.get(currentCosOffset + i);
                    float sinValue = sinArray.get(currentSinOffset + i);

                    float inputValueOne = input.get(inputTensorOffset + i);
                    float inputValueTwo = input.get(inputTensorOffset + halfHeadDim + i);

                    result.set(outputTensorOffset + i,
                            cosValue * inputValueOne - sinValue * inputValueTwo);
                    result.set(outputTensorOffset + i + halfHeadDim,
                            cosValue * inputValueTwo + sinValue * inputValueOne);
                }
            }
        }
    }

    public static void addRopeKernel(TaskGraph taskGraph, String name,
                                     @NonNull FloatArray input, @NonNull IntImmutableList inputShape,
                                     final int inputOffset,
                                     @NonNull FloatArray cosArray, final int cosOffset,
                                     @NonNull FloatArray sinArray, final int sinOffset,
                                     @NonNull FloatArray startPosition, int startPositionOffset,
                                     @NonNull FloatArray result, @NonNull IntImmutableList resultShape,
                                     final int resultOffset, int maxSequenceSize) {
        if (inputShape.size() != 4) {
            throw new IllegalArgumentException("Input shape must have rank 4 (batch size, sequence size, " +
                    "num heads, head dimension). Input shape: " + inputShape + ".");
        }

        if (!inputShape.equals(resultShape)) {
            throw new IllegalArgumentException("Input and result shapes must be the same. Input shape: " +
                    inputShape + ", result shape: " +
                    resultShape + ".");
        }

        if (cosArray.getSize() != inputShape.getInt(inputShape.size() - 1) * maxSequenceSize) {
            throw new IllegalArgumentException("Cos array size for RoPE must contain values for all " +
                    "possible positions in sequence. " +
                    "Cos array size: " + cosArray.getSize() + ", last dimension of the input shape: " +
                    inputShape.getInt(inputShape.size() - 1) + ", maximum sequence size: " + maxSequenceSize +
                    ". Expected size : " + inputShape.getInt(inputShape.size() - 1) * maxSequenceSize);
        }

        if (sinArray.getSize() != inputShape.getInt(inputShape.size() - 1) * maxSequenceSize) {
            throw new IllegalArgumentException("Sin array size for RoPE must contain values for all " +
                    "possible positions in sequence. " +
                    "Cos array size: " + cosArray.getSize() + ", last dimension of the input shape: " +
                    inputShape.getInt(inputShape.size() - 1) + ", maximum sequence size: " + maxSequenceSize +
                    ". Expected size : " + inputShape.getInt(inputShape.size() - 1) * maxSequenceSize);
        }

        var inputShapeArray = IntArray.fromArray(inputShape.toIntArray());
        taskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, inputShapeArray);

        taskGraph.task(TvmCommons.generateName(name),
                TvmTensorOperations::ropeKernel,
                input, inputShapeArray, inputOffset, cosArray, cosOffset, sinArray, sinOffset,
                startPosition, startPositionOffset, result, resultOffset, maxSequenceSize);
    }
}
