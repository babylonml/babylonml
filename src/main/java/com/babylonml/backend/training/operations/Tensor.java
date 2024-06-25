package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import it.unimi.dsi.fastutil.ints.IntImmutableList;

public final class Tensor {
    private final float[] data;
    private final IntImmutableList shape;

    public Tensor(float[] data, int[] shape) {
        var stride = TensorOperations.stride(shape);

        if (data.length != stride) {
            throw new IllegalArgumentException("Data length must be equal to the stride of the shape.");
        }

        this.data = data;
        this.shape = IntImmutableList.of(shape);
    }

    @SuppressWarnings("unused")
    public Tensor(int[] shape) {
        this(new float[TensorOperations.stride(shape)], shape);
    }

    public IntImmutableList getShape() {
        return shape;
    }

    public boolean isEmpty() {
        return data.length == 0;
    }

    public int size() {
        return data.length;
    }


    public float[] getData() {
        return data;
    }

    @SuppressWarnings("unused")
    public static Tensor fromVector(float[] data) {
        var shape = new int[]{data.length};
        return new Tensor(data, shape);
    }

    public static Tensor fromMatrix(float[][] data) {
        var shape = new int[]{data.length, data[0].length};
        var tensorData = new float[TensorOperations.stride(shape)];

        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, tensorData, i * data[i].length, data[i].length);
        }

        return new Tensor(tensorData, shape);
    }
}
