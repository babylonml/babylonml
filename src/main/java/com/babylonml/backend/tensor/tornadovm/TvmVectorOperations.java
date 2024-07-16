package com.babylonml.backend.tensor.tornadovm;

import org.jspecify.annotations.NonNull;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

public class TvmVectorOperations {
    static void addVectorToVectorKernel(FloatArray first, int firstOffset, FloatArray second, int secondOffset,
                                                FloatArray result, int resultOffset, int length) {
        for (@Parallel int i = 0; i < length; i++) {
            result.set(resultOffset + i, first.get(firstOffset + i) + second.get(secondOffset + i));
        }
    }

    static void addVectorToVectorKernel(FloatArray first, int firstOffset, ByteArray second, int secondOffset,
                                                FloatArray result, int resultOffset, int length) {
        for (@Parallel int i = 0; i < length; i++) {
            result.set(resultOffset + i, first.get(firstOffset + i) + second.get(secondOffset + i));
        }
    }

    static void copyVectorKernel(FloatArray source, int sourceOffset,
                                         FloatArray destination, int destinationOffset, int length) {
        for (@Parallel int i = 0; i < length; i++) {
            destination.set(destinationOffset + i, source.get(sourceOffset + i));
        }
    }

    static void copyVectorKernel(ByteArray source, int sourceOffset,
                                         ByteArray destination, int destinationOffset, int length) {
        for (@Parallel int i = 0; i < length; i++) {
            destination.set(destinationOffset + i, source.get(sourceOffset + i));
        }
    }

    public static void addVectorToVectorTask(TaskGraph graph, String name, @NonNull TornadoNativeArray firstArray, int firstOffset,
                                             @NonNull TornadoNativeArray secondArray, int secondOffset,
                                             @NonNull FloatArray resultArray, int resultOffset, int length) {
        switch (firstArray) {
            case FloatArray firstFloatArray when secondArray instanceof FloatArray secondFloatArray -> graph.task(name,
                    TvmVectorOperations::addVectorToVectorKernel,
                    firstFloatArray, firstOffset, secondFloatArray, secondOffset, resultArray, resultOffset, length);
            case FloatArray firstFloatArray when secondArray instanceof ByteArray secondByteArray -> graph.task(name,
                    TvmVectorOperations::addVectorToVectorKernel,
                    firstFloatArray, firstOffset, secondByteArray, secondOffset, resultArray, resultOffset, length);
            case ByteArray firstByteArray when secondArray instanceof FloatArray secondFloatArray -> graph.task(name,
                    TvmVectorOperations::addVectorToVectorKernel,
                    secondFloatArray, firstOffset, firstByteArray, secondOffset, resultArray, resultOffset, length);
            case null, default -> throw new IllegalArgumentException("Unsupported array types. " +
                    "First: " + firstArray.getClass() + " and second: " + secondArray.getClass());
        }
    }

    public static void addCopyVectorTask(TaskGraph graph, String name,
                                         @NonNull TornadoNativeArray source, int sourceOffset,
                                         @NonNull TornadoNativeArray destination, int destinationOffset, int length) {
        if (source instanceof FloatArray floatSource && destination instanceof FloatArray floatDestination) {
            graph.task(name,
                    TvmVectorOperations::copyVectorKernel,
                    floatSource, sourceOffset, floatDestination, destinationOffset, length);
        } else if (source instanceof ByteArray byteSource && destination instanceof ByteArray byteDestination) {
            graph.task(name,
                    TvmVectorOperations::copyVectorKernel,
                    byteSource, sourceOffset, byteDestination, destinationOffset, length);
        } else {
            throw new IllegalArgumentException("Unsupported array types. " +
                    "Source: " + source.getClass() + " and destination: " + destination.getClass());
        }

    }
}
