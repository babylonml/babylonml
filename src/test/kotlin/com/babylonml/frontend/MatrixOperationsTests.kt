package com.babylonml.frontend

import com.babylonml.backend.training.operations.Multiplication
import com.babylonml.frontend.MatrixTestHelper.validateVariable
import com.babylonml.frontend.MatrixTestHelper.validateOperation
import com.babylonml.frontend.MatrixTestHelper.toOpGraph
import com.tornadoml.cpu.SeedsArgumentsProvider
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource

class MatrixOperationsTests {

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testMatrixProduct(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val inner = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val matrix1 = Matrix.fill(rows, inner) { _, _ -> source.nextFloat() }
        val matrix2 = Matrix.fill(inner, columns) { _, _ -> source.nextFloat() }

        val matrix3 = matrix1 * matrix2
        assertEquals(MatrixDims(rows, columns), matrix3.dims)

        validateOperation(
            Multiplication::class, toOpGraph(matrix3),
            check = {
                assertEquals(rows, it.firstMatrixRows)
                assertEquals(inner, it.firstMatrixColumns)
                assertEquals(columns, it.secondMatrixColumns)
            },
            left = { validateVariable(it, matrix1) },
            right = { validateVariable(it, matrix2) }
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testMatrixProductDimensionsMismatch(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows1 = source.nextInt(1, 100)
        val columns1 = source.nextInt(1, 100)
        val rows2 = source.nextInt(1, 100)
        val columns2 = source.nextInt(1, 100)

        assumeTrue(columns1 != rows2)

        val matrix1 = Matrix.zeros(rows1, columns1)
        val matrix2 = Matrix.zeros(rows2, columns2)

        assertThrows(IllegalArgumentException::class.java) {
            matrix1 * matrix2
        }
    }
}
