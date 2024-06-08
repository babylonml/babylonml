package com.babylonml.frontend

import com.babylonml.backend.training.TrainingExecutionContext
import com.babylonml.backend.training.operations.*
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
        fun zeros(rows: Int, cols: Int): EagerMatrix =
            fill(rows, cols, 0.0f)

        /**
         * Creates a matrix of ones with the specified dimensions.
         */
        fun ones(rows: Int, cols: Int): EagerMatrix =
            fill(rows, cols, 1.0f)

        /**
         * Creates a matrix with the specified dimensions and initializes it with the specified function.
         */
        fun fill(rows: Int, cols: Int, init: (Int, Int) -> Float): EagerMatrix =
            EagerMatrix(FloatArray(rows * cols) { i -> init(i / cols, i % cols) }, MatrixDims(rows, cols))

        /**
         * Creates a matrix filled with the specified value with the specified dimensions.
         */
        fun fill(rows: Int, cols: Int, value: Float): EagerMatrix =
            fill(rows, cols) { _, _ -> value }

        /**
         * Creates an identity matrix of the specified size.
         */
        fun identity(size: Int): EagerMatrix =
            fill(size, size) { i, j -> if (i == j) 1.0f else 0.0f }

        /**
         * Creates a matrix from the specified data array.
         */
        fun from2DArray(data: Array<FloatArray>): EagerMatrix =
            EagerMatrix(flattenArray(data), MatrixDims(data.size, data[0].size))

        /**
         * Creates a matrix from the specified flat array.
         */
        fun fromFlatArray(rows: Int, cols: Int, data: FloatArray): EagerMatrix {
            require(data.size == rows * cols)
            return EagerMatrix(data, MatrixDims(rows, cols))
        }

        /**
         * Creates an N x 1 matrix representing a column vector from the specified data.
         */
        fun columnVector(vararg data: Float): EagerMatrix =
            EagerMatrix(data, MatrixDims(data.size, 1))

        /**
         * Creates a 1 x N matrix representing a row vector from the specified data.
         */
        fun rowVector(vararg data: Float): EagerMatrix =
            EagerMatrix(data, MatrixDims(1, data.size))


        private fun applyBinary(left: Matrix, right: Matrix, operation: MatrixOperation): Matrix {
            require(operation.type == MatrixOperationType.Binary)
            return MatrixExpression(left, right, operation.validateAndComputeDims(left.dims, right.dims), operation)
        }

        private fun applyBroadcastable(left: Matrix, right: Matrix, operation: MatrixOperation): Matrix {
            val resultingDims = MatrixDims.validateBroadcastDims(left.dims, right.dims)

            return applyBinary(
                left.broadcastRows(resultingDims.rows).broadcastColumns(resultingDims.columns),
                right.broadcastRows(resultingDims.rows).broadcastColumns(resultingDims.columns),
                operation
            )
        }

        private fun applyUnary(matrix: Matrix, operation: MatrixOperation): Matrix {
            require(operation.type == MatrixOperationType.Unary)
            return MatrixExpression(
                matrix,
                matrix,
                operation.validateAndComputeDims(matrix.dims, matrix.dims),
                operation
            )
        }
    }

    /**
     * Element-wise addition of two matrices. Broadcasts the matrices if necessary.
     */
    operator fun plus(other: Matrix): Matrix = applyBroadcastable(this, other, MatrixOperation.add)

    /**
     * Matrix multiplication of two matrices.
     */
    operator fun times(other: Matrix): Matrix = applyBinary(this, other, MatrixOperation.mul)

    /**
     * Element-wise multiplication of two matrices.
     */
    fun hadamardMul(other: Matrix): Matrix = applyBroadcastable(this, other, MatrixOperation.hadamardMul)

    /**
     * Applies the Gaussian Error Linear Unit (GeLU) function element-wise to the matrix.
     */
    fun gelu(): Matrix = applyUnary(this, MatrixOperation.geLU)

    /**
     * Broadcasts a one row matrix to the specified number of rows.
     */
    fun broadcastRows(rows: Int): Matrix = applyUnary(this, MatrixOperation.broadcastRows(rows))

    /**
     * Broadcasts a one column matrix to the specified number of columns.
     */
    fun broadcastColumns(cols: Int): Matrix = applyUnary(this, MatrixOperation.broadcastColumns(cols))
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
            for (j in 0 until dims.columns) {
                result.append(data[i * dims.columns + j])
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
data class MatrixDims(val rows: Int, val columns: Int) {
    companion object {
        fun validateBroadcastDims(left: MatrixDims, right: MatrixDims): MatrixDims {
            fun msg() =
                "Matrices are not broadcastable: ${left.rows}x${left.columns} and ${right.rows}x${right.columns}"
            require(left.rows == 1 || right.rows == 1 || left.rows == right.rows) { msg() }
            require(left.columns == 1 || right.columns == 1 || left.columns == right.columns) { msg() }

            return MatrixDims(max(left.rows, right.rows), max(left.columns, right.columns))
        }
    }
}

// INTERNAL API BELOW

/**
 * Internal API: Operation that can be applied to matrices.
 */
abstract class MatrixOperation(val type: MatrixOperationType) {
    /**
     * Validate the dimensions of the input matrices and compute the resulting dimensions of the output matrix.
     * For unary operations, the right dimensions should be ignored.
     * @throws IllegalArgumentException if the dimensions are invalid.
     */
    abstract fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims

    /**
     * Convert this matrix operation to a backend operation.
     * Happens during the materialization of the matrix expression.
     */
    abstract fun toBackendOperation(
        context: TrainingExecutionContext,
        left: Operation,
        right: Operation,
        leftDims: MatrixDims,
        rightDims: MatrixDims
    ): Operation

    companion object {

        /**
         * Matrix operation for element-wise addition.
         */
        val add = object : MatrixOperation(MatrixOperationType.Binary) {
            override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims {
                require(left == right)
                return left
            }

            override fun toBackendOperation(
                context: TrainingExecutionContext,
                left: Operation,
                right: Operation,
                leftDims: MatrixDims,
                rightDims: MatrixDims
            ): Operation = Add(context, leftDims.rows, leftDims.columns, left, right)
        }

        /**
         * Matrix operation for matrix multiplication.
         */
        val mul = object : MatrixOperation(MatrixOperationType.Binary) {
            override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims {
                require(left.columns == right.rows)
                return MatrixDims(left.rows, right.columns)
            }

            override fun toBackendOperation(
                context: TrainingExecutionContext,
                left: Operation,
                right: Operation,
                leftDims: MatrixDims,
                rightDims: MatrixDims
            ): Operation = Multiplication(context, leftDims.rows, leftDims.columns, rightDims.columns, left, right)
        }

        val hadamardMul = object : MatrixOperation(MatrixOperationType.Binary) {
            override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims {
                require(left == right)
                return left
            }

            override fun toBackendOperation(
                context: TrainingExecutionContext,
                left: Operation,
                right: Operation,
                leftDims: MatrixDims,
                rightDims: MatrixDims
            ): Operation = HadamardProduct(leftDims.rows, leftDims.columns, context, left, right)
        }

        /**
         * Matrix operation for applying the Gaussian Error Linear Unit (GeLU) function element-wise.
         */
        val geLU = object : MatrixOperation(MatrixOperationType.Unary) {
            override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims = left

            override fun toBackendOperation(
                context: TrainingExecutionContext,
                left: Operation,
                right: Operation,
                leftDims: MatrixDims,
                rightDims: MatrixDims
            ): Operation = GeLUFunction(leftDims.rows, leftDims.columns, context, left)
        }

        fun leakyLeRU(slope: Float) = object : MatrixOperation(MatrixOperationType.Unary) {
            override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims = left

            override fun toBackendOperation(
                context: TrainingExecutionContext,
                left: Operation,
                right: Operation,
                leftDims: MatrixDims,
                rightDims: MatrixDims
            ): Operation = LeakyLeRUFunction(leftDims.rows, leftDims.columns, slope, context, left)
        }

        /**
         * Matrix operation for broadcasting a one-row matrix to the specified number of rows.
         */
        fun broadcastRows(rows: Int) = object : MatrixOperation(MatrixOperationType.Unary) {
            override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims {
                require(left.rows == 1 || left.rows == rows)
                return MatrixDims(rows, left.columns)
            }

            override fun toBackendOperation(
                context: TrainingExecutionContext,
                left: Operation,
                right: Operation,
                leftDims: MatrixDims,
                rightDims: MatrixDims
            ): Operation =
                if (leftDims.rows == rows) left
                else BroadcastRows(rows, leftDims.columns, context, left)
        }

        /**
         * Matrix operation for broadcasting a one-column matrix to the specified number of columns.
         */
        fun broadcastColumns(cols: Int) = object : MatrixOperation(MatrixOperationType.Unary) {
            override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims {
                require(left.columns == 1 || left.columns == cols)
                return MatrixDims(left.rows, cols)
            }

            override fun toBackendOperation(
                context: TrainingExecutionContext,
                left: Operation,
                right: Operation,
                leftDims: MatrixDims,
                rightDims: MatrixDims
            ): Operation =
                if (leftDims.columns == cols) left
                else BroadcastColumns(leftDims.rows, cols, context, left)
        }

        /**
         * Combine two unary operations into a single unary operation. (not used for now)
         */
        fun combineUnary(first: MatrixOperation, second: MatrixOperation): MatrixOperation {
            require(first.type == MatrixOperationType.Unary && second.type == MatrixOperationType.Unary)

            return object : MatrixOperation(MatrixOperationType.Unary) {
                override fun validateAndComputeDims(left: MatrixDims, right: MatrixDims): MatrixDims {
                    val intermediateDims = first.validateAndComputeDims(left, left)
                    return second.validateAndComputeDims(intermediateDims, intermediateDims)
                }

                override fun toBackendOperation(
                    context: TrainingExecutionContext,
                    left: Operation,
                    right: Operation,
                    leftDims: MatrixDims,
                    rightDims: MatrixDims
                ): Operation {
                    val intermediate = first.toBackendOperation(context, left, left, leftDims, leftDims)
                    val intermediateDims = first.validateAndComputeDims(leftDims, leftDims)
                    return second.toBackendOperation(
                        context,
                        intermediate,
                        intermediate,
                        intermediateDims,
                        intermediateDims
                    )
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
