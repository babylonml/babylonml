package com.babylonml.backend.gemm;

import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class Gemm_03_Naive2 {

    public static void run(
            String graphName,
            FloatArray a, FloatArray b, FloatArray c,
            int m, int k, int n, float alpha, float beta
    ) {
        WorkerGrid workerGrid = new WorkerGrid2D(m, n);

        GridScheduler gridScheduler = new GridScheduler(graphName + ".t0", workerGrid);

        KernelContext context = new KernelContext();

        workerGrid.setLocalWork(16, 16, 1);

        TaskGraph taskGraph = new TaskGraph(graphName) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t0", Gemm_03_Naive2::runKernel, context, a, b, c, m, k, n, alpha, beta) //
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

    public static void runKernel(KernelContext context, FloatArray a, FloatArray b, FloatArray c, int m, int k, int n, float alpha, float beta) {
        int x = context.groupIdx * context.localGroupSizeX + context.localIdy;
        int y = context.groupIdy * context.localGroupSizeY + context.localIdx;
//        int x = context.globalIdx;
//        int y = context.globalIdy;

        if (x < m && y < n) {
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                sum += a.get(x * k + i) * b.get(i * n + y);
            }
            c.set(x * n + y, alpha * sum + beta * c.get(x * n + y));
        }
    }
}
