package com.babylonml.backend.gemm;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class Gemm_01_Cpu {

    public static void run(
            String graphName,
            FloatArray a, FloatArray b, FloatArray c,
            int m, int k, int n, float alpha, float beta
    ) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += a.get(i * k + l) * b.get(l * n + j);
                }
                c.set(i * n + j, alpha * sum + beta * c.get(i * n + j));
            }
        }
    }
}
