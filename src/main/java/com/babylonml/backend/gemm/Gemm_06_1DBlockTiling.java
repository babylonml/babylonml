package com.babylonml.backend.gemm;

import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class Gemm_06_1DBlockTiling {

    private static final int BM = 32;
    private static final int BN = BM; // = 32

    private static final int TM = 4;
    private static final int BK = BM / TM; // = 8

    public static void run(
            String graphName,
            FloatArray a, FloatArray b, FloatArray c,
            int m, int k, int n, float alpha, float beta
    ) {
        // gridDim: n / BN, m / BM
        // globalWorkSize = gridDim * localWorkSize
        // globalWorkSize.x = n / BN * BM * BN / TM = n * BM / TM
        // globalWorkSize.y = m / BM
        WorkerGrid2D workerGrid = new WorkerGrid2D(n * BM / TM, m / BM);
        workerGrid.setLocalWork(BM * BN / TM, 1, 1);

        GridScheduler gridScheduler = new GridScheduler(graphName + ".t0", workerGrid);

        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph(graphName) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t0", Gemm_06_1DBlockTiling::runKernel, context, a, b, c, m, k, n, alpha, beta) //
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
        final int cRow = context.groupIdy;
        final int cCol = context.groupIdx;

        final int threadCol = context.localIdx % BN;
        final int threadRow = context.localIdx / BN;

        final float[] as = context.allocateFloatLocalArray(BM * BK);
        final float[] bs = context.allocateFloatLocalArray(BK * BN);

        int aIndex = cRow * BM * k;
        int bIndex = cCol * BN;
        int cIndex = cRow * BM * n + cCol * BN;

        final int innerColA = context.localIdx % BK;
        final int innerRowA = context.localIdx / BK;
        final int innerColB = context.localIdx % BN;
        final int innerRowB = context.localIdx / BN;

        float[] threadResults = new float[TM];
        for (int i = 0; i < TM; i++) {
            threadResults[i] = 0.0f;
        }

        for (int blockIdx = 0; blockIdx < k; blockIdx += BK) {

            as[innerRowA * BK + innerColA] = a.get(aIndex + innerRowA * k + innerColA);
            bs[innerRowB * BN + innerColB] = b.get(bIndex + innerRowB * n + innerColB);

            context.localBarrier();

            aIndex += BK;
            bIndex += BK * n;

            for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
                float tmpB = bs[dotIdx * BN + threadCol];
                for (int resIdx = 0; resIdx < TM; resIdx++) {
                    threadResults[resIdx] += as[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
                }
            }

            context.localBarrier();
        }

        for (int resIdx = 0; resIdx < TM; resIdx++) {
            c.set(
                    cIndex + (threadRow * TM + resIdx) * n + threadCol,
                    alpha * threadResults[resIdx] + beta * c.get(cIndex + (threadRow * TM + resIdx) * n + threadCol)
            );
        }
    }
}
