package com.babylonml.frontend

import com.babylonml.backend.training.operations.AbstractOperation
import com.babylonml.backend.training.operations.BroadcastColumns
import com.babylonml.backend.training.operations.BroadcastRows
import com.babylonml.frontend.MatrixTestHelper.toOpGraph
import com.babylonml.frontend.MatrixTestHelper.validateOperation
import com.babylonml.frontend.MatrixTestHelper.validateVariable
import com.tornadoml.cpu.SeedsArgumentsProvider
import org.apache.commons.rng.simple.RandomSource
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assumptions.assumeFalse
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ArgumentsSource
import kotlin.reflect.KClass

/**
 * A test suite for binary operations that support broadcasting.
 */
abstract class BroadcastableBinaryOperationTestsSuite<Op : AbstractOperation>(
    val backendOpClass: KClass<Op>,
    val applyOp: (Matrix, Matrix) -> Matrix,
    val opDims: (Op) -> MatrixDims
) {

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testNoBroadcasting(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(1, 100)

        val matrix1 = Matrix.fill(rows, columns) { _, _ -> source.nextFloat() }
        val matrix2 = Matrix.fill(rows, columns) { _, _ -> source.nextFloat() }

        val matrix3 = applyOp(matrix1, matrix2)
        assertEquals(matrix1.dims, matrix3.dims)

        validateOperation(
            backendOpClass, toOpGraph(matrix3),
            check = { assertEquals(matrix3.dims, opDims(it)) },
            left = { validateVariable(it, matrix1) },
            right = { validateVariable(it, matrix2) }
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testBroadcastRowsLeft(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(2, 100)
        val columns = source.nextInt(1, 100)

        val matrix1 = Matrix.fill(1, columns) { _, _ -> source.nextFloat() }
        val matrix2 = Matrix.fill(rows, columns) { _, _ -> source.nextFloat() }

        val matrix3 = applyOp(matrix1, matrix2)
        assertEquals(matrix2.dims, matrix3.dims)

        validateOperation(
            backendOpClass, toOpGraph(matrix3),
            check = { assertEquals(matrix3.dims, opDims(it)) },
            left = {
                validateOperation(
                    BroadcastRows::class, it,
                    check = { assertEquals(matrix3.dims, MatrixDims(it.rows, it.columns)) },
                    left = { validateVariable(it, matrix1) },
                )
            },
            right = { validateVariable(it, matrix2) }
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testBroadcastRowsRight(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(2, 100)
        val columns = source.nextInt(1, 100)

        val matrix1 = Matrix.fill(rows, columns) { _, _ -> source.nextFloat() }
        val matrix2 = Matrix.fill(1, columns) { _, _ -> source.nextFloat() }

        val matrix3 = applyOp(matrix1, matrix2)
        assertEquals(matrix1.dims, matrix3.dims)

        validateOperation(
            backendOpClass, toOpGraph(matrix3),
            check = { assertEquals(matrix3.dims, opDims(it)) },
            left = { validateVariable(it, matrix1) },
            right = {
                validateOperation(
                    BroadcastRows::class, it,
                    check = { assertEquals(matrix3.dims, MatrixDims(it.rows, it.columns)) },
                    left = { validateVariable(it, matrix2) },
                )
            }
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testBroadcastColumnsLeft(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(2, 100)

        val matrix1 = Matrix.fill(rows, 1) { _, _ -> source.nextFloat() }
        val matrix2 = Matrix.fill(rows, columns) { _, _ -> source.nextFloat() }

        val matrix3 = applyOp(matrix1, matrix2)
        assertEquals(matrix2.dims, matrix3.dims)

        validateOperation(
            backendOpClass, toOpGraph(matrix3),
            check = { assertEquals(matrix3.dims, opDims(it)) },
            left = {
                validateOperation(
                    BroadcastColumns::class, it,
                    check = { assertEquals(matrix3.dims, MatrixDims(it.rows, it.columns)) },
                    left = { validateVariable(it, matrix1) },
                )
            },
            right = { validateVariable(it, matrix2) }
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testBroadcastColumnsRight(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(1, 100)
        val columns = source.nextInt(2, 100)

        val matrix1 = Matrix.fill(rows, columns) { _, _ -> source.nextFloat() }
        val matrix2 = Matrix.fill(rows, 1) { _, _ -> source.nextFloat() }

        val matrix3 = applyOp(matrix1, matrix2)
        assertEquals(matrix1.dims, matrix3.dims)

        validateOperation(
            backendOpClass, toOpGraph(matrix3),
            check = { assertEquals(matrix3.dims, opDims(it)) },
            left = { validateVariable(it, matrix1) },
            right = {
                validateOperation(
                    BroadcastColumns::class, it,
                    check = { assertEquals(matrix3.dims, MatrixDims(it.rows, it.columns)) },
                    left = { validateVariable(it, matrix2) },
                )
            }
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testBroadcastRowsAndColumns(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(2, 100)
        val columns = source.nextInt(2, 100)

        val matrix1 = Matrix.fill(1, columns) { _, _ -> source.nextFloat() }
        val matrix2 = Matrix.fill(rows, 1) { _, _ -> source.nextFloat() }

        val matrix3 = applyOp(matrix1, matrix2)
        assertEquals(MatrixDims(rows, columns), matrix3.dims)

        validateOperation(
            backendOpClass, toOpGraph(matrix3),
            check = { assertEquals(matrix3.dims, opDims(it)) },
            left = {
                validateOperation(
                    BroadcastRows::class, it,
                    check = { assertEquals(matrix3.dims, MatrixDims(it.rows, it.columns)) },
                    left = { validateVariable(it, matrix1) },
                )
            },
            right = {
                validateOperation(
                    BroadcastColumns::class, it,
                    check = { assertEquals(matrix3.dims, MatrixDims(it.rows, it.columns)) },
                    left = { validateVariable(it, matrix2) },
                )
            }
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testBroadcastColumnsAndRows(seed: Long) {
        val source = RandomSource.ISAAC.create(seed)

        val rows = source.nextInt(2, 100)
        val columns = source.nextInt(2, 100)

        val matrix1 = Matrix.fill(rows, 1) { _, _ -> source.nextFloat() }
        val matrix2 = Matrix.fill(1, columns) { _, _ -> source.nextFloat() }

        val matrix3 = applyOp(matrix1, matrix2)
        assertEquals(MatrixDims(rows, columns), matrix3.dims)

        validateOperation(
            backendOpClass, toOpGraph(matrix3),
            check = { assertEquals(matrix3.dims, opDims(it)) },
            left = {
                validateOperation(
                    BroadcastColumns::class, it,
                    check = { assertEquals(matrix3.dims, MatrixDims(it.rows, it.columns)) },
                    left = { validateVariable(it, matrix1) },
                )
            },
            right = {
                validateOperation(
                    BroadcastRows::class, it,
                    check = { assertEquals(matrix3.dims, MatrixDims(it.rows, it.columns)) },
                    left = { validateVariable(it, matrix2) },
                )
            }
        )
    }

    @ParameterizedTest
    @ArgumentsSource(SeedsArgumentsProvider::class)
    fun testDimensionMismatch(seed: Long) {

        val source = RandomSource.ISAAC.create(seed)

        val rows1 = source.nextInt(1, 100)
        val columns1 = source.nextInt(1, 100)

        val rows2 = source.nextInt(1, 100)
        val columns2 = source.nextInt(1, 100)

        // assume that the matrices are not broadcastable
        assumeFalse(
            (rows1 == 1 || rows2 == 1 || rows1 == rows2) && (columns1 == 1 || columns2 == 1 || columns1 == columns2)
        )

        val matrix1 = Matrix.zeros(rows1, columns1)
        val matrix2 = Matrix.zeros(rows2, columns2)

        assertThrows(IllegalArgumentException::class.java) {
            applyOp(matrix1, matrix2)
        }
    }
}
