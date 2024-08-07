package com.babylonml.backend.tensor.tornadovm;

import java.util.concurrent.atomic.AtomicLong;

public class TvmCommons {
    private static final AtomicLong idGenerator = new AtomicLong(0);

    public static String generateName(String name) {
        return name + "-" + idGenerator.getAndIncrement();
    }
}
