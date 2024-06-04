package com.babylonml.dsl.v1

import com.babylonml.backend.training.TrainingExecutionContext
import com.babylonml.backend.training.operations.Add
import com.babylonml.backend.training.operations.BroadcastColumns
import com.babylonml.backend.training.operations.BroadcastRows
import com.babylonml.backend.training.operations.GeLUFunction
import com.babylonml.backend.training.operations.Multiplication
import com.babylonml.backend.training.operations.Operation
import kotlin.math.max

/**
 * Matrix of float values.
 * This interface defines a computation graph that can be materialized into a concrete matrix.
 */
sealed interface Matrix {
    /**
     * Dimensions of the matrix.
     */
    val dims: MatrixDims

    /**
     * Materializes the computation into a concrete matrix.
     */
    fun materialize(): EagerMatrix

    companion object {
        /**
         * Creates a matrix of zeros with the specified dimensions.
         */
        fun zeros(rows: Int, cols: Int): EagerMatrix {
            return fill(rows, cols, 0.0f)
        }

        /**
         * Creates a matrix of ones with the specified dimensions.
         */
        fun ones(rows: Int, cols: Int): EagerMatrix {
            return fill(rows, cols, 1.0f)
        }

        /**
         * Creates a matrix filled with the specified value with the specified dimensions.
         */
        fun fill(rows: Int, cols: Int, value: Float): EagerMatrix {
            return EagerMatrix(FloatArray(rows * cols) { value }, MatrixDims(rows, cols))
        }

        /**
         * Creates a matrix from the specified data array.
         */
        fun of(data: Array<FloatArray>): EagerMatrix {
            return EagerMatrix(flattenArray(data), MatrixDims(data.size, data[0].size))
        }

        /**
         * Creates a N x 1 matrix representing a column vector from the specified data.
         */
        fun columnVector(vararg data: Float): EagerMatrix {
            return EagerMatrix(data, MatrixDims(data.size, 1))
        }

        /**
         * Creates a 1 x N matrix representing a row vector from the specified data.
         */
        fun rowVector(vararg data: Float): EagerMatrix {
            return EagerMatrix(data, MatrixDims(1, data.size))
        }
    }

    /**
     * Element-wise addition of two matrices. Broadcasts the matrices if necessary.
     */
    operator fun plus(other: Matrix): Matrix {
        val resultingDims = MatrixDims.validateBroadcastDims(this.dims, other.dims)

        return applyBinary(
            this.broadcastRows(resultingDims.rows).broadcastColumns(resultingDims.cols),
            other.broadcastRows(resultingDims.rows).broadcastColumns(resultingDims.cols),
            MatrixOperation.add
        )
    }

    /**
     * Matrix multiplication of two matrices.
     */
    operator fun times(other: Matrix): Matrix {
        return applyBinary(this, other, MatrixOperation.mul)
    }

    /**
     * Applies the Gaussian Error Linear Unit (GeLU) function element-wise to the matrix.
     */
    fun gelu(): Matrix {
        return applyUnary(this, MatrixOperation.gelu)
    }

    /**
     * Broadcasts a one row matrix to the specified number of rows.
     */
    fun broadcastRows(rows: Int): Matrix {
        return applyUnary(this, MatrixOperation.broadcastRows(rows))
    }

    /**
     * Broadcasts a one column matrix to the specified number of columns.
     */
    fun broadcastColumns(cols: Int): Matrix {
        return applyUnary(this, MatrixOperation.broadcastColumns(cols))
    }

    private fun applyBinary(left: Matrix, right: Matrix, operation: MatrixOperation): Matrix {
        require(operation.type == MatrixOperationType.Binary)
        return MatrixExpression(left, right, operation.validateAndComputeDims(left.dims, right.dims), operation)
    }

    private fun applyUnary(matrix: Matrix, operation: MatrixOperation): Matrix {
        require(operation.type == MatrixOperationType.Unary)
        return MatrixExpression(matrix, matrix, operation.validateAndComputeDims(matrix.dims, matrix.dims), operation)
    }
}

/**
 * A matrix whose elements are already computed.
 */
class EagerMatrix(
    val data: FloatArray,
    override val dims: MatrixDims
) : Matrix {
    override fun materialize(): EagerMatrix = this

    /**
     * Converts the matrix to a pretty string representation.
     */
    fun contentToString(): String {
        val result = StringBuilder("[\n")
        for (i in 0 until dims.rows) {
            result.append(" [ ")
            for (j in 0 until dims.cols) {
                result.append(data[i * dims.cols + j])
                result.append(", ")
            }
            result.append("]\n")
        }
        return result.append("]").toString()

    }
}

/**
 * An expression that can be materialized into a concrete matrix and consists of a binary or unary operation
 * applied to one or two matrices.
 */
class MatrixExpression(
    val left: Matrix,
    val right: Matrix, // for unary operations this will be equal to left
    override val dims: MatrixDims,
    val op: MatrixOperation
) : Matrix {
    override fun materialize(): EagerMatrix = EagerMatrix(
        data = MatrixEngine.compute(this),
        dims = this.dims
    )
}

/**
 * Dimensions of a matrix.
 */
data class MatrixDims(val rows: Int, val cols: Int) {
    companion object {
        fun validateBroadcastDims(left: MatrixDims, right: MatrixDims): MatrixDims {
            require(left.rows == 1 || right.rows == 1 || left.rows == right.rows)
            require(left.cols == 1 || right.cols == 1 || left.cols == right.cols)
            return MatrixDims(max(left.rows, right.rows), max(left.cols, right.cols))
        }
    }
}

// INTERNAL API BELOW

abstract class MatrixOperation(val type: MatrixOperationType) {
    abstract fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims

    abstract fun toOperation(
        context: TrainingExecutionContext,
        left: Operation,
        right: Operation,
        leftDims: MatrixDims,
        rightDims: MatrixDims
    ): Operation

    companion object {
        val add = object : MatrixOperation(MatrixOperationType.Binary) {
            override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims {
                require(left == right)
                return left
            }

            override fun toOperation(
                context: TrainingExecutionContext,
                left: Operation,
                right: Operation,
                leftDims: MatrixDims,
                rightDims: MatrixDims
            ): Operation = Add(context, leftDims.rows, leftDims.cols, left, right)

        }

        val mul = object : MatrixOperation(MatrixOperationType.Binary) {
            override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims {
                require(left.cols == right.rows)
                return MatrixDims(left.rows, right.cols)
            }

            override fun toOperation(
                context: TrainingExecutionContext,
                left: Operation,
                right: Operation,
                leftDims: MatrixDims,
                rightDims: MatrixDims
            ): Operation = Multiplication(context, leftDims.rows, leftDims.cols, rightDims.cols, left, right)
        }

        val gelu = object : MatrixOperation(MatrixOperationType.Unary) {
            override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims = left

            override fun toOperation(
                context: TrainingExecutionContext,
                left: Operation,
                right: Operation,
                leftDims: MatrixDims,
                rightDims: MatrixDims
            ): Operation = GeLUFunction(leftDims.rows, leftDims.cols, context, left)
        }

        fun broadcastRows(rows: Int) = object : MatrixOperation(MatrixOperationType.Unary) {
            override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims {
                require(left.rows == 1 || left.rows == rows)
                return MatrixDims(rows, left.cols)
            }

            override fun toOperation(
                context: TrainingExecutionContext,
                left: Operation,
                right: Operation,
                leftDims: MatrixDims,
                rightDims: MatrixDims
            ): Operation =
                if (leftDims.rows == rows) left
                else BroadcastRows(rows, leftDims.cols, context, left)
        }

        fun broadcastColumns(cols: Int) = object : MatrixOperation(MatrixOperationType.Unary) {
            override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims {
                require(left.cols == 1 || left.cols == cols)
                return MatrixDims(left.rows, cols)
            }

            override fun toOperation(
                context: TrainingExecutionContext,
                left: Operation,
                right: Operation,
                leftDims: MatrixDims,
                rightDims: MatrixDims
            ): Operation =
                if (leftDims.cols == cols) left
                else BroadcastColumns(leftDims.rows, cols, context, left)
        }

        fun combineUnary(first: MatrixOperation, second: MatrixOperation): MatrixOperation {
            require(first.type == MatrixOperationType.Unary && second.type == MatrixOperationType.Unary)

            return object : MatrixOperation(MatrixOperationType.Unary) {
                override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims {
                    val intermediateDims = first.validateAndComputeDims(left, left)
                    return second.validateAndComputeDims(intermediateDims, intermediateDims)
                }

                override fun toOperation(
                    context: TrainingExecutionContext,
                    left: Operation,
                    right: Operation,
                    leftDims: MatrixDims,
                    rightDims: MatrixDims
                ): Operation {
                    val intermediate = first.toOperation(context, left, left, leftDims, leftDims)
                    val intermediateDims = first.validateAndComputeDims(leftDims, leftDims)
                    return second.toOperation(context, intermediate, intermediate, intermediateDims, intermediateDims)
                }
            }
        }
    }
}

enum class MatrixOperationType {
    Binary,
    Unary
}

private fun flattenArray(matrix: Array<FloatArray>): FloatArray {
    val rows = matrix.size
    val cols = matrix[0].size
    val result = FloatArray(rows * cols)

    for (i in matrix.indices) {
        require(matrix[i].size == cols)
        for (j in matrix[0].indices) {
            result[i * cols + j] = matrix[i][j]
        }
    }

    return result
}
