package com.babylonml.backend.tornadovm;

import com.babylonml.backend.common.CommonTensorOperations;
import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;
import uk.ac.manchester.tornado.api.TaskGraph;
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

        taskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inputShapeArray, outputShapeArray);
        taskGraph.task(TvmCommons.generateName(prefix + "-broadcastKernel"),
                TvmTensorOperations::broadcastKernel,
                input, inputShapeArray, inputOffset, output, outputOffset, outputStride, outputShapeArray);
    }

    private static void broadcastKernel(@NonNull FloatArray input, @NonNull IntArray inputShape, int inputOffset,
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
}
