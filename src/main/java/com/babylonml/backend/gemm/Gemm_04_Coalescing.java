package com.babylonml.backend.gemm;

import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class Gemm_04_Coalescing {

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
                .task("t0", Gemm_04_Coalescing::runKernel, context, a, b, c, m, k, n, alpha, beta) //
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
        final int x = context.groupIdx * BLOCK_SIZE + (context.localIdx / BLOCK_SIZE);
        final int y = context.groupIdy * BLOCK_SIZE + (context.localIdx % BLOCK_SIZE);

        if (x < m && y < n) {
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                sum += a.get(x * k + i) * b.get(i * n + y);
            }
            c.set(x * n + y, alpha * sum + beta * c.get(x * n + y));
        }
    }
}
