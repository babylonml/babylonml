package com.babylonml.backend.tornadovm;

import com.babylonml.backend.common.CommonTensorOperations;
import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class TvmTensorOperations {
    public static void broadcast(@NonNull FloatArray input, int inputOffset, @NonNull IntImmutableList inputShape,
                                 @NonNull FloatArray output, int outputOffset, @NonNull IntImmutableList outputShape,
                                 TaskGraph taskGraph,
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
        inputShape = CommonTensorOperations.extendShapeTill(inputShape, outputShape.size());

        //adjust broadcastTillRank to the new shape
        broadcastTillRank += inputShape.size() - prevSize;

        if (outputShape.size() > broadcastTillRank) {
            var modifiedShape = new int[outputShape.size()];
            outputShape.getElements(0, modifiedShape, 0, broadcastTillRank);
            inputShape.getElements(broadcastTillRank, modifiedShape, broadcastTillRank,
                    outputShape.size() - broadcastTillRank);

            outputShape = IntImmutableList.of(modifiedShape);
        }

        if (CommonTensorOperations.isNotBroadcastCompatible(inputShape, outputShape)) {
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

        var inputStride = CommonTensorOperations.stride(inputShape);
        if (batchRank == -1) {
            taskGraph.task(TvmCommons.generateName("copyVectorForBroadcast"), () ->
                    TvmVectorOperations.copyVector(input, inputOffset, output, outputOffset, inputStride));
            return;
        }

        var currentRank = 0;
        var outputStride = CommonTensorOperations.stride(outputShape);

        copyAndBroadcastDimension(input, inputOffset, inputStride / inputShape.getInt(currentRank),
                inputShape, output, outputOffset,
                outputStride / outputShape.getInt(currentRank),
                outputShape,
                currentRank, batchRank, taskGraph);
    }

    private static void copyAndBroadcastDimension(@NonNull FloatArray input, int inputOffset, int inputStrideWidth,
                                                  @NonNull IntImmutableList inputShape,
                                                  @NonNull FloatArray output,
                                                  int outputOffset, int outputStrideWidth,
                                                  @NonNull IntImmutableList outputShape,
                                                  int currentRank, int batchRank, TaskGraph taskGraph) {
        assert currentRank <= batchRank;

        if (currentRank == batchRank) {
            taskGraph.task(TvmCommons.generateName("copyVectorForBroadcast"), () ->
                    TvmVectorOperations.copyVector(input, inputOffset, output, outputOffset, inputStrideWidth));

            assert inputStrideWidth == outputStrideWidth;
            if (inputShape.getInt(currentRank) != outputShape.getInt(currentRank)) {
                assert inputShape.getInt(currentRank) == 1;

                duplicateDimension(output, outputOffset, outputStrideWidth, outputShape, currentRank, taskGraph);
            }
        } else {
            var inputDimension = inputShape.getInt(currentRank);
            var outputDimension = outputShape.getInt(currentRank);

            for (int i = 0; i < inputDimension; i++) {
                copyAndBroadcastDimension(input, inputOffset + i * inputStrideWidth,
                        inputStrideWidth / inputShape.getInt(currentRank + 1), inputShape,
                        output, outputOffset + i * outputStrideWidth,
                        outputStrideWidth / outputShape.getInt(currentRank + 1), outputShape,
                        currentRank + 1, batchRank, taskGraph);
            }

            if (inputDimension != outputDimension) {
                assert inputDimension == 1;
                duplicateDimension(output, outputOffset, outputStrideWidth, outputShape, currentRank, taskGraph);
            }
        }
    }

    private static void duplicateDimension(@NonNull FloatArray output, int outputOffset,
                                           int outputStrideWidth, @NonNull IntImmutableList outputShape,
                                           int currentRank, TaskGraph taskGraph) {
        var repeat = outputShape.getInt(currentRank);
        var outputIndex = outputOffset + outputStrideWidth;

        var batchSize = 1;
        int i = 1;

        while (i + batchSize <= repeat) {
            var batchWidth = batchSize * outputStrideWidth;

            var copyOutputIndex = outputIndex;
            taskGraph.task(TvmCommons.generateName("copyVectorForBroadcast"), () ->
                    TvmVectorOperations.copyVector(output, outputOffset, output, copyOutputIndex, batchWidth));

            outputIndex += batchWidth;
            i += batchSize;

            batchSize <<= 1;
        }

        if (i < repeat) {
            batchSize = repeat - i;

            var copyOutputIndex = outputIndex;
            var copyBatchSize = batchSize;

            taskGraph.task(TvmCommons.generateName("copyVectorForBroadcast"), () ->
                    TvmVectorOperations.copyVector(output, outputOffset, output, copyOutputIndex,
                            copyBatchSize * outputStrideWidth));
        }
    }
}
