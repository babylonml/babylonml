package com.babylonml.backend.common;

import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;

public class CommonTensorOperations {
    public static int stride(@NonNull IntImmutableList shape) {
        int stride = 1;
        for (int i = 0; i < shape.size(); i++) {
            stride *= shape.getInt(i);
        }
        return stride;
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
            throw new IllegalArgumentException("Size must be greater than or equal to the size of the shape to be extended");
        }

        return shapeToBroadcast;
    }

    public static @NonNull IntImmutableList cutShapeTill(@NonNull IntImmutableList shapeToDecrease, int size) {
        if (size < shapeToDecrease.size()) {
            var newInputShape = new int[size];

            for (int i = 0; i < shapeToDecrease.size() - size; i++) {
                if (shapeToDecrease.getInt(i) != 1) {
                    throw new IllegalArgumentException("Shape to decrease must have 1s in the dimensions to cut, but got " +
                            shapeToDecrease + ". ");
                }
            }

            shapeToDecrease.getElements(shapeToDecrease.size() - size, newInputShape, 0,
                    size);

            return new IntImmutableList(newInputShape);
        } else if (size > shapeToDecrease.size()) {
            throw new IllegalArgumentException("Size must be less than or equal to the size of the shape to be cut");
        }

        return shapeToDecrease;
    }

    public static boolean isNotBroadcastCompatible(@NonNull IntImmutableList firstShape,
                                                   @NonNull IntImmutableList secondShape) {
        for (int i = 0; i < secondShape.size(); i++) {
            if (firstShape.getInt(i) != secondShape.getInt(i) &&
                    firstShape.getInt(i) != 1 && secondShape.getInt(i) != 1) {
                return true;
            }
        }

        return false;
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
}
