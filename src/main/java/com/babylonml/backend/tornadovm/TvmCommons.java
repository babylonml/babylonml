package com.babylonml.backend.tornadovm;

import java.util.concurrent.atomic.AtomicLong;

public class TvmCommons {
    private static AtomicLong idGenerator = new AtomicLong(0);

    public static String generateName(String name) {
        return name + "-" + idGenerator.getAndIncrement();
    }
}
