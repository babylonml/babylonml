package com.babylonml.backend.gemm

import com.babylonml.backend.TvmFloatArray
import org.apache.commons.rng.simple.RandomSource
import java.security.SecureRandom

object GemmDemo {
    @JvmStatic
    fun main(args: Array<String>) {
//        val m = 4
//        val k = 4
//        val n = 4
//
//        val m = 1024
//        val k = 512
//        val n = 2048

//        val m = 2048
//        val k = 1024
//        val n = 4096
//
        val m = 4096
        val k = 4096
        val n = 4096
//
        val source = RandomSource.ISAAC.create(SecureRandom().generateSeed(8))

        val alpha = 1.0F
        val beta = 0.0F

        val operations = linkedMapOf(
//            "cpu" to Gemm_01_Cpu::run,
            "parallel" to Gemm_02_Parallel::run,
            "naive" to Gemm_03_Naive::run,
            "naive2" to Gemm_03_Naive2::run,
            "coalescing" to Gemm_04_Coalescing::run,
            "coalescing2" to Gemm_04_Coalescing2::run,
            "sharedmem" to Gemm_05_SharedMemory::run,
            "1dblocktiling" to Gemm_06_1DBlockTiling::run,
            "2dblocktiling" to Gemm_07_2DBlockTiling::run,
        )

        // random
        val A = FloatArray(m * k) { source.nextFloat() }
        val B = FloatArray(k * n) { source.nextFloat() }

        var expected: TvmFloatArray? = null;

        for ((name, operation) in operations) {
            val itCount = if (name == "cpu") 1 else 10

            for (it in 0 until itCount) {
                if (it == 0) {
                    println("==========")
                    println("Running $name")
                }

                val a = TvmFloatArray.fromArray(A)
                val b = TvmFloatArray.fromArray(B)
                val c = TvmFloatArray(m * n)

                val start = System.nanoTime()
                operation("s-$name-$it", a, b, c, m, k, n, alpha, beta)
                val end = System.nanoTime()

                println("Time: ${(end - start) / 1e6} ms")

                if (expected == null) {
                    expected = c
                } else {
                    for (i in 0 until m * n) {
                        if (!(expected[i] isCloseTo c[i])) {
                            println("Mismatch at index $i: expected ${expected[i]}, got ${c[i]}")
                            break
                        }
                    }
                }
                if (it == itCount - 1) {
                    println()
                }
            }
        }
    }

    infix fun Float.isCloseTo(other: Float): Boolean = isCloseTo(other, 0.0001f)

    fun Float.isCloseTo(other: Float, tolerance: Float): Boolean {
        return kotlin.math.abs(this - other) <= tolerance
    }
}
