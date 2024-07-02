package com.babylonml.backend.tornadovm;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class TvmVectorOperations {
    public static void addVectorToVector(FloatArray first, int firstOffset, FloatArray second, int secondOffset,
                                         FloatArray result, int resultOffset, int length) {
        for (@Parallel int i = 0; i < length; i++) {
            result.set(resultOffset + i, first.get(firstOffset + i) + second.get(secondOffset + i));
        }
    }

    public static void copyVector(FloatArray source, int sourceOffset, FloatArray destination, int destinationOffset, int length) {
        for (@Parallel int i = 0; i < length; i++) {
            destination.set(destinationOffset + i, source.get(sourceOffset + i));
        }
    }
}
