package com.babylonml.backend.gemm;

import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class Gemm_07_2DBlockTiling {

    private static final int BK = 4;
    private static final int TM = 8;
    private static final int TN = 8;

    private static final int BM = 128;
    private static final int BN = 128;

    public static void run(
            String graphName,
            FloatArray a, FloatArray b, FloatArray c,
            int m, int k, int n, float alpha, float beta
    ) {
        // gridDim: n / BN, m / BM
        // globalWorkSize = gridDim * localWorkSize
        // globalWorkSize.x = (n / BN) * BM * BN / (TM * TN) = n * BM / (TM * TN)
        // globalWorkSize.y = m / BM
        WorkerGrid2D workerGrid = new WorkerGrid2D(n * BM / (TM * TN), m / BM);
        workerGrid.setLocalWork((BM * BN) / (TM * TN), 1, 1);

        GridScheduler gridScheduler = new GridScheduler(graphName + ".t0", workerGrid);

        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph(graphName) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t0", Gemm_07_2DBlockTiling::runKernel, context, a, b, c, m, k, n, alpha, beta) //
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
        int cRow = context.groupIdy;
        int cCol = context.groupIdx;

        int totalResultsBlockTile = BM * BN;
        int numThreadsBlockTile = totalResultsBlockTile / (TM * TN);

        int threadCol = context.localIdx % (BN / TN);
        int threadRow = context.localIdx / (BN / TN);

        float[] as = context.allocateFloatLocalArray(BM * BK);
        float[] bs = context.allocateFloatLocalArray(BK * BN);

        int ai = cRow * BM * k;
        int bi = cCol * BN;
        int ci = cRow * BM * n + cCol * BN;

        int innerRowA = context.localIdx / BK;
        int innerColA = context.localIdx % BK;
        int strideA = numThreadsBlockTile / BK;

        int innerRowB = context.localIdx / BN;
        int innerColB = context.localIdx % BN;
        int strideB = numThreadsBlockTile / BN;

        float[] threadResults = new float[TM * TN];
        for (int i = 0; i < TM * TN; i++) {
            threadResults[i] = 0.0f;
        }

        float[] regM = new float[TM];
        float[] regN = new float[TN];

        for (int bkIdx = 0; bkIdx < k; bkIdx += BK) {
            for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
                as[(innerRowA + loadOffset) * BK + innerColA] = a.get(ai + (innerRowA + loadOffset) * k + innerColA);
            }

            for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
                bs[(innerRowB + loadOffset) * BN + innerColB] = b.get(bi + (innerRowB + loadOffset) * n + innerColB);
            }
            context.localBarrier();

            ai += BK;
            bi += BK * n;

            for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                for (int i = 0; i < TM; i++) {
                    regM[i] = as[(threadRow * TM + i) * BK + dotIdx];
                }

                for (int i = 0; i < TN; i++) {
                    regN[i] = bs[dotIdx * BN + threadCol * TN + i];
                }

                for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
                    for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                        threadResults[resIdxM * TN + resIdxN] =
                                threadResults[resIdxM * TN + resIdxN] + regM[resIdxM] * regN[resIdxN];
                    }
                }
            }

            context.localBarrier();
        }

        for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
            for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                c.set(
                        ci + (threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN,
                        alpha * threadResults[resIdxM * TN + resIdxN] + beta * c.get(ci + (threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN)
                );
            }
        }
    }
}
