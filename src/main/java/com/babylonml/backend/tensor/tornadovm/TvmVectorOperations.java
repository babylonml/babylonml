package com.babylonml.backend.tensor.tornadovm;

import org.jspecify.annotations.NonNull;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

public class TvmVectorOperations {
    static void addVectorToVectorKernel(FloatArray first, int firstOffset, FloatArray second, int secondOffset,
                                        FloatArray result, int resultOffset, KernelContext kernelContext) {

        int i = kernelContext.globalIdx;
        result.set(resultOffset + i, first.get(firstOffset + i) + second.get(secondOffset + i));
    }

    static void addVectorToVectorKernel(FloatArray first, int firstOffset, ByteArray second, int secondOffset,
                                        FloatArray result, int resultOffset, KernelContext kernelContext) {
        int i = kernelContext.globalIdx;
        result.set(resultOffset + i, first.get(firstOffset + i) + second.get(secondOffset + i));
    }

    static void copyVectorKernel(FloatArray source, int sourceOffset,
                                 FloatArray destination, int destinationOffset, KernelContext kernelContext) {
        int i = kernelContext.globalIdx;
        destination.set(destinationOffset + i, source.get(sourceOffset + i));
    }

    static void copyVectorKernel(ByteArray source, int sourceOffset,
                                 ByteArray destination, int destinationOffset, KernelContext kernelContext) {
        int i = kernelContext.globalIdx;
        destination.set(destinationOffset + i, source.get(sourceOffset + i));
    }

    public static void addVectorToVectorTask(@NonNull TaskGraph graph, @NonNull String name,
                                             @NonNull GridScheduler gridScheduler,
                                             @NonNull TornadoNativeArray firstArray, int firstOffset,
                                             @NonNull TornadoNativeArray secondArray, int secondOffset,
                                             @NonNull FloatArray resultArray, int resultOffset, int length) {

        var kernelContext = new KernelContext();
        var taskName = TvmCommons.generateName(name);
        switch (firstArray) {
            case FloatArray firstFloatArray when secondArray instanceof FloatArray secondFloatArray ->
                    graph.task(taskName,
                            TvmVectorOperations::addVectorToVectorKernel,
                            firstFloatArray, firstOffset, secondFloatArray, secondOffset, resultArray, resultOffset,
                            kernelContext);
            case FloatArray firstFloatArray when secondArray instanceof ByteArray secondByteArray ->
                    graph.task(taskName,
                            TvmVectorOperations::addVectorToVectorKernel,
                            firstFloatArray, firstOffset, secondByteArray, secondOffset, resultArray, resultOffset,
                            kernelContext);
            case ByteArray firstByteArray when secondArray instanceof FloatArray secondFloatArray ->
                    graph.task(taskName,
                            TvmVectorOperations::addVectorToVectorKernel,
                            secondFloatArray, firstOffset, firstByteArray, secondOffset, resultArray, resultOffset,
                            kernelContext);
            default -> throw new IllegalArgumentException("Unsupported array types. " +
                    "First: " + firstArray.getClass() + " and second: " + secondArray.getClass());
        }

        TvmCommons.initMapWorkerGrid1D(length, graph, taskName, gridScheduler);
    }

    public static void addCopyVectorTask(@NonNull TaskGraph graph, @NonNull String name,
                                         @NonNull GridScheduler gridScheduler,
                                         @NonNull TornadoNativeArray source, int sourceOffset,
                                         @NonNull TornadoNativeArray destination, int destinationOffset, int length) {
        var kernelContext = new KernelContext();
        if (source instanceof FloatArray floatSource && destination instanceof FloatArray floatDestination) {
            graph.task(name,
                    TvmVectorOperations::copyVectorKernel,
                    floatSource, sourceOffset, floatDestination, destinationOffset, kernelContext);
        } else if (source instanceof ByteArray byteSource && destination instanceof ByteArray byteDestination) {
            graph.task(name,
                    TvmVectorOperations::copyVectorKernel,
                    byteSource, sourceOffset, byteDestination, destinationOffset, kernelContext);
        } else {
            throw new IllegalArgumentException("Unsupported array types. " +
                    "Source: " + source.getClass() + " and destination: " + destination.getClass());
        }

        TvmCommons.initMapWorkerGrid1D(length, graph, name, gridScheduler);
    }
}
