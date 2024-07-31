package com.babylonml.backend.tensor.tornadovm;

import com.babylonml.backend.tensor.common.CommonTensorOperations;
import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.*;

public class TvmTensorOperations {
    public static void addBroadcastTask(TaskGraph taskGraph, String prefix,
                                        @NonNull GridScheduler gridScheduler,
                                        @NonNull FloatArray input,
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
                    prefix + "-copyVectorForBroadcast", gridScheduler, input, inputOffset,
                    output, outputOffset, inputStride);
            return;
        }

        var outputStride = CommonTensorOperations.stride(outputShape);
        var inputShapeArray = IntArray.fromArray(inputShape.toIntArray());
        var outputShapeArray = IntArray.fromArray(outputShape.toIntArray());

        taskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, inputShapeArray, outputShapeArray);

        var kernelContext = new KernelContext();

        var taskName = TvmCommons.generateName(prefix + "-broadcastKernel");
        taskGraph.task(taskName,
                TvmTensorOperations::broadcastKernel,
                input, inputShapeArray, inputOffset, output, outputOffset, outputShapeArray, kernelContext);
        TvmCommons.initMapWorkerGrid1D(outputStride, taskGraph, taskName, gridScheduler);
    }

    static void broadcastKernel(@NonNull FloatArray input, @NonNull IntArray inputShape, int inputOffset,
                                @NonNull FloatArray output, int outputOffset,
                                @NonNull IntArray outputShape,
                                KernelContext kernelContext) {
        final int outputShapeSize = outputShape.getSize();
        final int outputIndex = kernelContext.globalIdx;

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

    static void ropeKernel(@NonNull FloatArray input, final int inputOffset, int batchSize,
                           int sequenceSize, int numberOfHeads, int headDimension,
                           @NonNull FloatArray cosArray, final int cosOffset, @NonNull FloatArray sinArray,
                           final int sinOffset, @NonNull IntArray startPosition, int startPositionOffset,
                           @NonNull FloatArray result, final int resultOffset, @NonNull KernelContext kernelContext) {
        //cos/sin array is tensor with shape [positions, headDimension/ 2]
        //input is tensor with shape [batchSize, sequenceSize, numHeads, headDimension]
        //we are interested in head dimension only, so we distill all dimension into the
        //[batchSize * maxSequenceSize, numberOfHeads, headDimension] tensors
        // and apply RoPE to each [headDimension] tensor.
        final int sequenceStepSize = numberOfHeads * headDimension;
        final int batchStepSize = sequenceStepSize * sequenceSize;

        final int halfHeadDim = headDimension / 2;

        final int batchSequenceIteration = kernelContext.globalIdx;

        final int batchIndex = batchSequenceIteration / sequenceSize;
        final int sequenceIndex = batchSequenceIteration % sequenceSize;
        final int batchOffset = batchIndex * batchStepSize;

        final int position = startPosition.get(startPositionOffset) + sequenceIndex;
        final int currentCosOffset = halfHeadDim * position + cosOffset;
        final int currentSinOffset = halfHeadDim * position + sinOffset;

        final int sequenceOffset = batchOffset + sequenceIndex * sequenceStepSize;

        final int h = kernelContext.globalIdy;

        final int commonOffset = sequenceOffset + h * headDimension;
        final int inputTensorOffset = inputOffset + commonOffset;
        final int outputTensorOffset = resultOffset + commonOffset;


        final int i = kernelContext.globalIdz;

        float cosValue = cosArray.get(currentCosOffset + i);
        float sinValue = sinArray.get(currentSinOffset + i);

        float inputValueOne = input.get(inputTensorOffset + i);
        float inputValueTwo = input.get(inputTensorOffset + halfHeadDim + i);

        result.set(outputTensorOffset + i,
                cosValue * inputValueOne - sinValue * inputValueTwo);
        result.set(outputTensorOffset + i + halfHeadDim,
                cosValue * inputValueTwo + sinValue * inputValueOne);
    }

    public static void addRopeKernel(TaskGraph taskGraph, String name,
                                     @NonNull GridScheduler gridScheduler,
                                     @NonNull FloatArray input, @NonNull IntImmutableList inputShape,
                                     final int inputOffset,
                                     @NonNull FloatArray cosArray, final int cosOffset,
                                     @NonNull FloatArray sinArray, final int sinOffset,
                                     @NonNull IntArray startPosition, int startPositionOffset,
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

        final int batchSize = inputShape.getInt(0);
        final int sequenceSize = inputShape.getInt(1);
        final int numberOfHeads = inputShape.getInt(2);
        final int headDimension = inputShape.getInt(3);

        if (sequenceSize > maxSequenceSize) {
            throw new IllegalArgumentException("Sequence size must be less or equal to max sequence size. " +
                    "Sequence size: " + sequenceSize + ", max sequence size: " + maxSequenceSize + ".");
        }
        if (sequenceSize % 2 != 0) {
            throw new IllegalArgumentException("Sequence size must be even. Sequence size: " + sequenceSize + ".");
        }

        if (headDimension % 2 != 0) {
            throw new IllegalArgumentException("Head dimension must be even. Head dimension: " + headDimension + ".");
        }


        final int batchSequenceIterations = batchSize * sequenceSize;
        final int halfHeadDim = headDimension / 2;

        var kernelContext = new KernelContext();
        var taskName = TvmCommons.generateName(name);
        taskGraph.task(taskName,
                TvmTensorOperations::ropeKernel,
                input, inputOffset, batchSize, sequenceSize, numberOfHeads, headDimension, cosArray, cosOffset,
                sinArray, sinOffset, startPosition, startPositionOffset,
                result, resultOffset, kernelContext);
        TvmCommons.initMapWorkerGrid3D(batchSequenceIterations, numberOfHeads, halfHeadDim, taskGraph, taskName, gridScheduler);
    }

    private static void sumKernel(@NonNull FloatArray inputTensor,
                                  int inputOffset,
                                  int tensorLength,
                                  @NonNull FloatArray result,
                                  int resultOffset,
                                  int resultTensorLength,
                                  @NonNull KernelContext context) {
        //context.globalIdx is the index of the current element in the tensor
        //context.globalIdy is the index of the current tensor in the batch
        var currentInputOffset = inputOffset + context.globalIdy * tensorLength + context.globalIdx;
        var localSnippet = context.allocateFloatLocalArray(TvmCommons.DEFAULT_REDUCE_LOCAL_ALLOCATION);

        localSnippet[context.localIdx] = inputTensor.get(currentInputOffset);

        if (context.globalIdx + context.globalGroupSizeX < tensorLength) {
            var reminderOffset = currentInputOffset + context.globalGroupSizeX;
            var valueReminder = inputTensor.get(reminderOffset);
            localSnippet[context.localIdx] += valueReminder;
        }

        context.localBarrier();

        //reduce the data
        int currentLocalGroupSizeX = context.localGroupSizeX;
        for (int s = currentLocalGroupSizeX >>> 1; s > 0; currentLocalGroupSizeX = s, s = currentLocalGroupSizeX >>> 1) {
            if (context.localIdx < s) {
                localSnippet[context.localIdx] += localSnippet[context.localIdx + s];
            }

            if (context.localIdx == 0 && ((currentLocalGroupSizeX & 1) == 1)) {
                localSnippet[0] += localSnippet[currentLocalGroupSizeX - 1];
            }

            context.localBarrier();
        }

        //write the result
        if (context.localIdx == 0) {
            var currentResultOffset = resultOffset +
                    context.globalIdy * resultTensorLength + context.globalIdx / context.localGroupSizeX;
            result.set(currentResultOffset, localSnippet[0]);
        }
    }


    private static void squareSumKernel(@NonNull FloatArray inputTensor,
                                        int inputOffset,
                                        int tensorLength,
                                        @NonNull FloatArray result,
                                        int resultOffset,
                                        int resultTensorLength,
                                        @NonNull KernelContext context) {
        //context.globalIdx is the index of the current element in the tensor
        //context.globalIdy is the index of the current tensor in the batch
        var currentInputOffset = inputOffset + context.globalIdy * tensorLength + context.globalIdx;
        var localSnippet = context.allocateFloatLocalArray(TvmCommons.DEFAULT_REDUCE_LOCAL_ALLOCATION);

        float value = inputTensor.get(currentInputOffset);
        localSnippet[context.localIdx] = value * value;

        if (context.globalIdx + context.globalGroupSizeX < tensorLength) {
            var reminderOffset = currentInputOffset + context.globalGroupSizeX;
            var valueReminder = inputTensor.get(reminderOffset);
            localSnippet[context.localIdx] += valueReminder * valueReminder;
        }

        context.localBarrier();

        //reduce the data
        int currentLocalGroupSizeX = context.localGroupSizeX;
        for (int s = currentLocalGroupSizeX >>> 1; s > 0; currentLocalGroupSizeX = s, s = currentLocalGroupSizeX >>> 1) {
            if (context.localIdx < s) {
                localSnippet[context.localIdx] += localSnippet[context.localIdx + s];
            }

            if (context.localIdx == 0 && ((currentLocalGroupSizeX & 1) == 1)) {
                localSnippet[0] += localSnippet[currentLocalGroupSizeX - 1];
            }

            context.localBarrier();
        }

        //write the result
        if (context.localIdx == 0) {
            var currentResultOffset = resultOffset +
                    context.globalIdy * resultTensorLength + context.globalIdx / context.localGroupSizeX;
            result.set(currentResultOffset, localSnippet[0]);
        }
    }

    public static void addSquareSumKernel(@NonNull TaskGraph taskGraph, String name,
                                          @NonNull GridScheduler gridScheduler,
                                          @NonNull FloatArray inputTensor,
                                          int inputOffset,
                                          int tensorCount,
                                          int tensorLength,
                                          @NonNull FloatArray resultTensor,
                                          int resultOffset) {
        var kernelContext = new KernelContext();
        var taskName = TvmCommons.generateName(name);

        var resultTensorLength = TvmCommons.initReduceWorkerGrid(tensorLength, tensorCount,
                TvmCommons.DEFAULT_REDUCE_LOCAL_ALLOCATION, taskGraph, taskName, gridScheduler);

        taskGraph.task(taskName, TvmTensorOperations::squareSumKernel,
                inputTensor, inputOffset, tensorLength,
                resultTensor, resultOffset, resultTensorLength, kernelContext);

        if (resultTensorLength > 1) {
            addSumKernel(taskGraph, name + "-sqr-sum-tail", gridScheduler, resultTensor, resultOffset, tensorCount,
                    resultTensorLength, resultTensor, resultOffset);
        }
    }

    public static void addSumKernel(@NonNull TaskGraph taskGraph, String name,
                                    @NonNull GridScheduler gridScheduler,
                                    @NonNull FloatArray inputTensor,
                                    int inputOffset,
                                    int tensorCount,
                                    int tensorLength,
                                    @NonNull FloatArray resultTensor,
                                    int resultOffset) {
        var currentTensorLength = tensorLength;
        do {
            var kernelContext = new KernelContext();
            var resultTensorLength = TvmCommons.initReduceWorkerGrid(currentTensorLength, tensorCount,
                    TvmCommons.DEFAULT_REDUCE_LOCAL_ALLOCATION, taskGraph, name, gridScheduler);

            var taskName = TvmCommons.generateName(name);
            taskGraph.task(taskName, TvmTensorOperations::sumKernel,
                    inputTensor, inputOffset, tensorLength,
                    resultTensor, resultOffset, resultTensorLength, kernelContext);

            currentTensorLength = resultTensorLength;
            inputOffset = resultOffset;
            inputTensor = resultTensor;
        } while (currentTensorLength > 1);
    }

    static void rmsNormF32WeightsKernel(@NonNull FloatArray input, int inputOffset,
                                        int tensorLength,
                                        @NonNull FloatArray weights, int weightsOffset,
                                        @NonNull FloatArray result, int resultOffset,
                                        FloatArray squareSum, int squareSumOffset,
                                        float epsilon, KernelContext kernelContext) {
        final int iteration = kernelContext.globalIdy;
        final int currenInputOffset = inputOffset + iteration * tensorLength;
        final int currentResultOffset = resultOffset + iteration * tensorLength;
        final int currentSquareSumOffset = squareSumOffset + iteration;

        final int tensorIndex = kernelContext.globalIdx;
        result.set(currentResultOffset + tensorIndex,
                input.get(tensorIndex + currenInputOffset) * weights.get(weightsOffset + tensorIndex)
                        /
                        TornadoMath.sqrt(squareSum.get(currentSquareSumOffset) / tensorLength + epsilon));
    }

    static void rmsNormF16WeightsKernel(@NonNull FloatArray input, int inputOffset, int tensorLength,
                                        @NonNull HalfFloatArray weights, int weightsOffset,
                                        @NonNull FloatArray result, int resultOffset,
                                        FloatArray squareSum, int squareSumOffset, float epsilon,
                                        KernelContext kernelContext) {
        final int iteration = kernelContext.globalIdy;
        final int currenInputOffset = inputOffset + iteration * tensorLength;
        final int currentResultOffset = resultOffset + iteration * tensorLength;
        final int currentSquareSumOffset = squareSumOffset + iteration;

        final int tensorIndex = kernelContext.globalIdx;
        result.set(currentResultOffset + tensorIndex,
                input.get(tensorIndex + currenInputOffset) * weights.get(weightsOffset + tensorIndex).getFloat32()
                        /
                        TornadoMath.sqrt(squareSum.get(currentSquareSumOffset) / tensorLength + epsilon));
    }

    static void rmsNormI8WeightsKernel(@NonNull FloatArray input,
                                       int inputOffset,
                                       int tensorLength,
                                       @NonNull ByteArray weights, int weightsOffset,
                                       @NonNull FloatArray result, int resultOffset, FloatArray squareSum,
                                       int squareSumOffset, float epsilon, KernelContext kernelContext) {
        final int iteration = kernelContext.globalIdy;

        final int currenInputOffset = inputOffset + iteration * tensorLength;
        final int currentResultOffset = resultOffset + iteration * tensorLength;
        final int currentSquareSumOffset = squareSumOffset + iteration;

        final int tensorIndex = kernelContext.globalIdx;
        result.set(currentResultOffset + tensorIndex,
                input.get(tensorIndex + currenInputOffset) * weights.get(weightsOffset + tensorIndex)
                        /
                        TornadoMath.sqrt(squareSum.get(currentSquareSumOffset) / tensorLength + epsilon));

    }

    public static void addRMSNormKernel(@NonNull TaskGraph taskGraph,
                                        @NonNull String name,
                                        @NonNull GridScheduler gridScheduler,
                                        @NonNull FloatArray input, @NonNull IntImmutableList inputShape, int inputOffset,
                                        @NonNull FloatArray squareResult, int squareResultOffset,
                                        @NonNull TornadoNativeArray weights, int weightsOffset,
                                        @NonNull FloatArray result, int resultOffset,
                                        float epsilon) {
        var meanSquareKernelName = "squareSumKernel-" + name;
        var tensorCount = 1;

        for (int i = 0; i < inputShape.size() - 1; i++) {
            tensorCount *= inputShape.getInt(i);
        }

        var tensorLength = inputShape.getInt(inputShape.size() - 1);
        addSquareSumKernel(taskGraph, meanSquareKernelName, gridScheduler,
                input, inputOffset, tensorCount, tensorLength, squareResult,
                squareResultOffset);

        var kernelContext = new KernelContext();
        var taskName = TvmCommons.generateName(name);
        switch (weights) {
            case FloatArray floatWeights -> taskGraph.task(taskName,
                    TvmTensorOperations::rmsNormF32WeightsKernel,
                    input, inputOffset, tensorLength, floatWeights, weightsOffset, result, resultOffset,
                    squareResult, squareResultOffset, epsilon, kernelContext);
            case HalfFloatArray halfWeights -> taskGraph.task(taskName,
                    TvmTensorOperations::rmsNormF16WeightsKernel,
                    input, inputOffset, tensorLength, halfWeights, weightsOffset, result, resultOffset,
                    squareResult, squareResultOffset, epsilon, kernelContext);
            case ByteArray byteWeights -> taskGraph.task(taskName,
                    TvmTensorOperations::rmsNormI8WeightsKernel,
                    input, inputOffset, tensorLength, byteWeights, weightsOffset, result, resultOffset,
                    squareResult, squareResultOffset, epsilon, kernelContext);
            default -> throw new IllegalArgumentException("Unsupported type of weights: " + weights.getClass());
        }

        TvmCommons.initMapWorkerGrid2D(tensorLength, tensorCount, taskGraph, taskName, gridScheduler);
    }
}
