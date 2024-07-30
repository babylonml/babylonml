package com.babylonml.backend.tensor.tornadovm;

import org.apache.commons.math3.util.ArithmeticUtils;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

import java.util.concurrent.atomic.AtomicLong;

public class TvmCommons {
    private static final int[] MAX_WORKGROUP_DIMENSIONS;

    static {
        var maxDimensions = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice().getDeviceMaxWorkgroupDimensions();
        MAX_WORKGROUP_DIMENSIONS = new int[maxDimensions.length];
        for (int i = 0; i < maxDimensions.length; i++) {
            MAX_WORKGROUP_DIMENSIONS[i] = (int) maxDimensions[i];
        }
    }


    private static final AtomicLong idGenerator = new AtomicLong(0);

    public static String generateName(String name) {
        return name + "-" + idGenerator.getAndIncrement();
    }

    public static void initMapWorkerGrid1D(int dim, TaskGraph taskGraph, String taskName, GridScheduler gridScheduler) {
        var workerGrid = new WorkerGrid1D(dim);

        workerGrid.setLocalWork(ArithmeticUtils.gcd(dim, MAX_WORKGROUP_DIMENSIONS[0]), 1, 1);
        gridScheduler.setWorkerGrid(taskGraph.getTaskGraphName() + "." + taskName, workerGrid);
    }

    public static void initMapWorkerGrid2D(int firstDim, int secondDim, TaskGraph taskGraph,
                                           String taskName, GridScheduler gridScheduler) {
        var workerGrid = new WorkerGrid2D(firstDim, secondDim);
        workerGrid.setLocalWork(ArithmeticUtils.gcd(firstDim, 16),
                ArithmeticUtils.gcd(secondDim, 16), 1);
        gridScheduler.setWorkerGrid(taskGraph.getTaskGraphName() + "." + taskName, workerGrid);
    }

    public static void initMapWorkerGrid3D(int firstDim, int secondDim, int thirdDim, TaskGraph taskGraph,
                                           String taskName, GridScheduler gridScheduler) {
        var workerGrid = new WorkerGrid3D(firstDim, secondDim, thirdDim);
        workerGrid.setLocalWork(ArithmeticUtils.gcd(firstDim, 8),
                ArithmeticUtils.gcd(secondDim, 8), ArithmeticUtils.gcd(thirdDim, 8));
        gridScheduler.setWorkerGrid(taskGraph.getTaskGraphName() + "." + taskName, workerGrid);
    }
}
