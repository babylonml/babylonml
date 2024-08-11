package com.babylonml.backend.gemm;

import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

import java.util.Arrays;

public class CoalescingDemo {

    private static final int BLOCK_SIZE = 2;

    public static void main(String[] args) {
        final int m = 16;
        final int k = 16;
        final int n = 8;
        final float alpha = 1.0F;
        final float beta = 0.0F;


        final float[] a = new float[m * k];
        final float[] b = new float[k * n];
        final float[] c1 = new float[m * n];
        final float[] c2 = new float[m * n];
        final float[] c3 = new float[m * n];

        for (int i = 0; i < m * k; i++) {
            a[i] = (float) Math.random();
        }
        for (int i = 0; i < k * n; i++) {
            b[i] = (float) Math.random();
        }

        printMatrix("A", a, m, k);
        printMatrix("B", b, k, n);

        runOnCpu(a, b, c1, m, k, n, alpha, beta);
        printMatrix("C (CPU)", c1, m, n);

        runNaive(a, b, c2, m, k, n, alpha, beta);
        printMatrix("C (Tornado naive)", c2, m, n);

        runCoalescing(a, b, c3, m, k, n, alpha, beta);
        printMatrix("C (Tornado coalescing)", c3, m, n);
    }

    private static void printMatrix(String name, float[] matrix, int rows, int cols) {
        System.out.println("Matrix " + name + ":");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.print(matrix[i * cols + j] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void runOnCpu(
            float[] a, float[] b, float[] c,
            int m, int k, int n, float alpha, float beta
    ) {
        for (int x = 0; x < m; x++) {
            for (int y = 0; y < n; y++) {
                float sum = 0.0f;
                for (int i = 0; i < k; i++) {
                    sum += a[x * k + i] * b[i * n + y];
                }
                c[x * n + y] = alpha * sum + beta * c[x * n + y];
            }
        }
    }

    private static void runNaive(float[] a, float[] b, float[] c, int m, int k, int n, float alpha, float beta) {
        WorkerGrid workerGrid = new WorkerGrid2D(m, n);
        workerGrid.setLocalWork(BLOCK_SIZE, BLOCK_SIZE, 1);

        System.out.println("Workgroups (naive): " + Arrays.toString(workerGrid.getNumberOfWorkgroups()));

        GridScheduler gridScheduler = new GridScheduler("naive" + ".t0", workerGrid);

        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph("naive") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t0", CoalescingDemo::naiveKernel, context, a, b, c, m, k, n, alpha, beta) //
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

    public static void naiveKernel(KernelContext context, float[] a, float[] b, float[] c, int m, int k, int n, float alpha, float beta) {

        int ordinal = context.globalGroupSizeX * context.globalIdy + context.globalIdx;

        int x = context.globalIdx;
        int y = context.globalIdy;

        if (x < m && y < n) {
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                sum += a[x * k + i] * b[i * n + y];
            }
            c[x * n + y] = alpha * sum + beta * c[x * n + y];
//             c[x * n + y] = ordinal;
        }
    }


    private static void runCoalescing(float[] a, float[] b, float[] c, int m, int k, int n, float alpha, float beta) {
        FixedGrid2D workerGrid = new FixedGrid2D(m, n);
        workerGrid.setLocalWork(BLOCK_SIZE * BLOCK_SIZE, 1, 1);
//        workerGrid.setNumberOfWorkgroups(m / BLOCK_SIZE, n / BLOCK_SIZE);

        System.out.println("Workgroups (coalescing): " + Arrays.toString(workerGrid.getNumberOfWorkgroups()));

        GridScheduler gridScheduler = new GridScheduler("coalescing.t0", workerGrid);

        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph("coalescing") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t0", CoalescingDemo::coalescingKernel, context, a, b, c, m, k, n, m / BLOCK_SIZE, n / BLOCK_SIZE, alpha, beta) //
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


    public static void coalescingKernel(
            KernelContext context, float[] a, float[] b, float[] c,
            int m, int k, int n,
            int mTiles, int nTiles,
            float alpha, float beta
    ) {
        int ordinal = context.globalGroupSizeX * context.globalIdy + context.globalIdx;

        int blockOrdinal = ordinal / (BLOCK_SIZE * BLOCK_SIZE);
        int localOrdinal = ordinal % (BLOCK_SIZE * BLOCK_SIZE);

        int blockX = blockOrdinal / nTiles;
        int blockY = blockOrdinal % nTiles;

        int x = blockX * BLOCK_SIZE + (localOrdinal / BLOCK_SIZE);
        int y = blockY * BLOCK_SIZE + (localOrdinal % BLOCK_SIZE);


        if (x < m && y < n) {
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                sum += a[x * k + i] * b[i * n + y];
            }
            c[x * n + y] = alpha * sum + beta * c[x * n + y];
            // c[x * n + y] = ordinal;
        }
    }

    public static class FixedGrid2D extends WorkerGrid2D {
        public FixedGrid2D(int x, int y) {
            super(x, y);
        }

        public void setNumberOfWorkgroups(int x, int y) {
            this.numOfWorkgroups = new long[]{x, y, 1};
        }
    }

}
