package com.tornadoml.cpu

import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import java.nio.ByteBuffer
import java.security.SecureRandom
import kotlin.math.pow

class AdamTests {
    @Test
    fun testAdam() {
        val securesRandom = SecureRandom()

        for (n in 0..9) {
            val bBuffer = ByteBuffer.wrap(securesRandom.generateSeed(8))
            val seed = bBuffer.getLong()

            println("testAdam seed: $seed")
            val source = RandomSource.ISAAC.create(seed)

            var weightsM = FloatMatrix(source.nextInt(100), source.nextInt(100))
            var weightsV = FloatMatrix(weightsM.rows, weightsM.cols)

            var biasesM = FloatVector(source.nextInt(100))
            var biasesV = FloatVector(biasesM.size)

            var biases = FloatVector(biasesM.size)
            var weights = FloatMatrix(weightsM.rows, weightsM.cols)

            val gradientsW = FloatMatrix(weightsM.rows, weightsM.cols)
            val gradientsB = FloatVector(biasesM.size)


            weights.fillRandom(source)
            biases.fillRandom(source)

            val betta1 = 0.9f
            val betta2 = 0.999f
            val learningRate = 0.001f
            val epsilon = 0.00000001f

            var biasesTested = biases.copy()
            var weightsTested = weights.copy()

            val optimizer = AdamOptimizer(weights.size, biases.size)

            for (iteration in 1..100) {
                gradientsW.fillRandom(source)
                gradientsB.fillRandom(source)

                weightsM = (weightsM * betta1) + (gradientsW * (1 - betta1))
                weightsV = (weightsV * betta2) + (gradientsW.hadamardMul(gradientsW) * (1 - betta2))

                biasesM = (biasesM * betta1) + (gradientsB * (1 - betta1))
                biasesV = (biasesV * betta2) + (gradientsB * gradientsB) * (1 - betta2)

                val weightsMCorrected = weightsM / (1 - betta1.pow(iteration))
                val weightsVCorrected = weightsV / (1 - betta2.pow(iteration))

                val biasesMCorrected = biasesM / (1 - betta1.pow(iteration))
                val biasesVCorrected = biasesV / (1 - betta2.pow(iteration))

                weights -= (learningRate / (weightsVCorrected.sqrt() + epsilon)).hadamardMul(weightsMCorrected)
                biases -= (learningRate / (biasesVCorrected.sqrt() + epsilon)) * biasesMCorrected

                val weightsFlat = weightsTested.toFlatArray()
                val biasesFlat = biasesTested.toArray()

                optimizer.optimize(
                    weightsFlat, gradientsW.toFlatArray(), weightsFlat.size, biasesFlat, gradientsB.toArray(),
                    biasesFlat.size, learningRate
                )

                Assertions.assertArrayEquals(
                    weights.toFlatArray(), weightsFlat, 0.0001f, "iteration: " +
                            iteration + " weights are not equal"
                )
                Assertions.assertArrayEquals(
                    biases.toArray(), biasesFlat, 0.0001f, "iteration: " +
                            iteration + " biases are not equal"
                )

                biasesTested = biases.copy()
                weightsTested = weights.copy()
            }
        }
    }
}