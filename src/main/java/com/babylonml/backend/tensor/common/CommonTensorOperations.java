package com.babylonml.backend.tensor.common;

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

  public static @NonNull IntImmutableList extendShapeTill(
      @NonNull IntImmutableList shapeToBroadcast, int size) {
    if (size > shapeToBroadcast.size()) {
      var newInputShape = new int[size];

      for (int i = 0; i < size - shapeToBroadcast.size(); i++) {
        newInputShape[i] = 1;
      }

      shapeToBroadcast.getElements(0, newInputShape, size - shapeToBroadcast.size(),
          shapeToBroadcast.size());

      return new IntImmutableList(newInputShape);
    } else if (size < shapeToBroadcast.size()) {
      throw new IllegalArgumentException(
          "Size must be greater than or equal to the size of the shape to be extended");
    }

    return shapeToBroadcast;
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
}
