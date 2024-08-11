package com.babylonml.backend.gemm;

import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class Gemm_05_SharedMemory {

    private static final int BLOCK_SIZE = 16;

    public static void run(
            String graphName,
            FloatArray a, FloatArray b, FloatArray c,
            int m, int k, int n, float alpha, float beta
    ) {
        WorkerGrid2D workerGrid = new WorkerGrid2D(m * BLOCK_SIZE, n / BLOCK_SIZE);
        workerGrid.setLocalWork(BLOCK_SIZE * BLOCK_SIZE, 1, 1);

        GridScheduler gridScheduler = new GridScheduler(graphName + ".t0", workerGrid);

        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph(graphName) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t0", Gemm_05_SharedMemory::runKernel, context, a, b, c, m, k, n, alpha, beta) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(immutableTaskGraph);
        executor.withGridScheduler(gridScheduler);

        executor.execute();
        try {
            executor.close();
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }
    }

    public static void runKernel(
            KernelContext context, FloatArray a, FloatArray b, FloatArray c,
            int m, int k, int n,
            float alpha, float beta
    ) {
        final int cRow = context.groupIdx;
        final int cCol = context.groupIdy;

        final float[] as = context.allocateFloatLocalArray(BLOCK_SIZE * BLOCK_SIZE);
        final float[] bs = context.allocateFloatLocalArray(BLOCK_SIZE * BLOCK_SIZE);

        final int threadCol = context.localIdx % BLOCK_SIZE;
        final int threadRow = context.localIdx / BLOCK_SIZE;

        int aIndex = cRow * BLOCK_SIZE * k;
        int bIndex = cCol * BLOCK_SIZE;
        int cIndex = cRow * BLOCK_SIZE * n + cCol * BLOCK_SIZE;

        float tmp = 0.0f;
        for (int blockIdx = 0; blockIdx < k; blockIdx += BLOCK_SIZE) {
            as[threadRow * BLOCK_SIZE + threadCol] = a.get(aIndex + threadRow * k + threadCol);
            bs[threadRow * BLOCK_SIZE + threadCol] = b.get(bIndex + threadRow * n + threadCol);

            context.localBarrier();

            aIndex += BLOCK_SIZE;
            bIndex += BLOCK_SIZE * n;

            for (int i = 0; i < BLOCK_SIZE; i++) {
                tmp += as[threadRow * BLOCK_SIZE + i] * bs[i * BLOCK_SIZE + threadCol];
            }

            context.localBarrier();
        }

        c.set(cIndex + threadRow * n + threadCol, alpha * tmp + beta * c.get(cIndex + threadRow * n + threadCol));
    }
}
