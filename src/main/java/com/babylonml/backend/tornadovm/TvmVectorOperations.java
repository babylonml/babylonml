package com.babylonml.backend.tornadovm;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class TvmVectorOperations {
    private static void addVectorToVectorKernel(FloatArray first, int firstOffset, FloatArray second, int secondOffset,
                                                FloatArray result, int resultOffset, int length) {
        for (@Parallel int i = 0; i < length; i++) {
            result.set(resultOffset + i, first.get(firstOffset + i) + second.get(secondOffset + i));
        }
    }

    private static void copyVectorKernel(FloatArray source, int sourceOffset, FloatArray destination, int destinationOffset, int length) {
        for (@Parallel int i = 0; i < length; i++) {
            destination.set(destinationOffset + i, source.get(sourceOffset + i));
        }
    }

    public static void addVectorToVectorTask(TaskGraph graph, String name, FloatArray firstArray, int firstOffset,
                                             FloatArray secondArray, int secondOffset,
                                             FloatArray resultArray, int resultOffset, int length) {
        graph.task(name,
                TvmVectorOperations::addVectorToVectorKernel,
                firstArray, firstOffset, secondArray, secondOffset, resultArray, resultOffset, length);
    }

    public static void addCopyVectorTask(TaskGraph graph, String name, FloatArray source, int sourceOffset,
                                         FloatArray destination, int destinationOffset, int length) {
        graph.task(name,
                TvmVectorOperations::copyVectorKernel,
                source, sourceOffset, destination, destinationOffset, length);
    }
}
