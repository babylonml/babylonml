package com.babylonml.backend.gemm;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class Gemm_02_Parallel {

    public static void run(
            String graphName,
            FloatArray a, FloatArray b, FloatArray c,
            int m, int k, int n, float alpha, float beta
    ) {

        final var taskGraph = new TaskGraph(graphName)
                .transferToDevice(
                        DataTransferMode.FIRST_EXECUTION,
                        a, b, c, m, k, n, alpha, beta
                )
                .task(
                        "t0", Gemm_02_Parallel::doRun, a, b, c, m, k, n, alpha, beta
                )
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        final var immutableTaskGraph = taskGraph.snapshot();
        final var executionPlan = new TornadoExecutionPlan(immutableTaskGraph);

        executionPlan.execute();
        try {
            executionPlan.close();
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }
    }

    private static void doRun(
            FloatArray a, FloatArray b, FloatArray c,
            int m, int k, int n, float alpha, float beta
    ) {
        for (@Parallel int i = 0; i < m; i++) {
            for (@Parallel int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += a.get(i * k + l) * b.get(l * n + j);
                }
                c.set(i * n + j, alpha * sum + beta * c.get(i * n + j));
            }
        }
    }
}
