package com.babylonml.backend.examples.mnist;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

public final class MNISTLoader {
    public static float[][] loadMNISTImages() throws IOException {
        try (var stream = MNISTLoader.class.getClassLoader().getResourceAsStream("mnist/train-images.idx3-ubyte")) {
            return loadImages(stream);
        }
    }

    public static int[] loadMNISTLabels() throws IOException {
        try (var stream = MNISTLoader.class.getClassLoader().getResourceAsStream("mnist/train-labels.idx1-ubyte")) {
            return loadLabels(stream);
        }
    }

    @SuppressWarnings("unused")
    public static float[][] loadMNISTTestImages() throws IOException {
        try (var stream = MNISTLoader.class.getClassLoader().getResourceAsStream("mnist/t10k-images.idx3-ubyte")) {
            return loadImages(stream);
        }
    }

    @SuppressWarnings("unused")
    public static int[] loadMNISTTestLabels() throws IOException {
        try (var stream = MNISTLoader.class.getClassLoader().getResourceAsStream("mnist/t10k-labels.idx1-ubyte")) {
            return loadLabels(stream);
        }
    }

    private static int[] loadLabels(InputStream stream) throws IOException {
        Objects.requireNonNull(stream, "Resource not found");
        try (var dataStream = new DataInputStream(stream)) {
            var magic = dataStream.readInt();
            if (magic != 0x801) {
                throw new IOException("Invalid magic number: " + magic);
            }

            var count = dataStream.readInt();
            var labels = new int[count];

            for (int i = 0; i < count; i++) {
                labels[i] = dataStream.read() & 0xff;
                assert labels[i] >= 0 && labels[i] <= 9;
            }

            return labels;
        }
    }

    private static float[][] loadImages(InputStream stream) throws IOException {
        Objects.requireNonNull(stream, "Resource not found");

        try (var dataStream = new DataInputStream(stream)) {
            var magic = dataStream.readInt();
            if (magic != 0x803) {
                throw new IOException("Invalid magic number: " + magic);
            }

            var count = dataStream.readInt();
            var rows = dataStream.readInt();
            var cols = dataStream.readInt();

            if (rows != 28 || cols != 28) {
                throw new IOException("Invalid image size: " + rows + "x" + cols);
            }

            var images = new float[count][rows * cols];
            for (int i = 0; i < images.length; i++) {
                for (int j = 0; j < rows * cols; j++) {
                    images[i][j] = (dataStream.read() & 0xff) / 255.0f;
                }
            }

            return images;
        }
    }
}
