package com.babylonml.backend.gemm;

import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class GemmTiling {

    private final static int TS = 32;

    public static void run(
            String graphName,
            FloatArray a, FloatArray b, FloatArray c,
            int m, int k, int n, float alpha, float beta
    ) {

        WorkerGrid workerGrid = new WorkerGrid2D(m, n);

        GridScheduler gridScheduler = new GridScheduler(graphName + ".t0", workerGrid);

        KernelContext context = new KernelContext();

        workerGrid.setLocalWork(32, 32, 1);

        TaskGraph taskGraph = new TaskGraph(graphName) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t0", GemmTiling::matrixMultiplication, context, a, b, c, m, k, n, alpha, beta) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(immutableTaskGraph);
        executor.withGridScheduler(gridScheduler);

        executor.execute();
    }

    public static void matrixMultiplication(KernelContext context, FloatArray a, FloatArray b, FloatArray c, int m, int k, int n, float alpha, float beta) {
//        int row = context.localIdx;
//        int col = context.localIdy;
//        int globalRow = TS * context.groupIdx + row;
//        int globalCol = TS * context.groupIdy + col;
//
//        float[] aSub = context.allocateFloatLocalArray(TS * TS);
//        float[] bSub = context.allocateFloatLocalArray(TS * TS);
//
//        float sum = 0;
//
//        // Loop over all tiles
//        int numTiles = k / TS;
//        for (int t = 0; t < numTiles; t++) {
//
//            // Load one tile of A and B into local memory
//            int tiledRow = TS * t + row;
//            int tiledCol = TS * t + col;
//            aSub[col * TS + row] = a.get(tiledCol * size + globalRow);
//            bSub[col * TS + row] = b.get(globalCol * size + tiledRow);
//
//            // Synchronise to make sure the tile is loaded
//            context.localBarrier();
//
//            // Perform the computation for a single tile
//            for (int k = 0; k < TS; k++) {
//                sum += aSub[k * TS + row] * bSub[col * TS + k];
//            }
//            // Synchronise before loading the next tile
//            context.globalBarrier();
//        }
//
//        // Store the final result in C
//        C.set((globalCol * size) + globalRow, sum);
    }
}
