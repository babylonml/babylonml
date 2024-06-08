package com.babylonml.frontend

import com.babylonml.backend.training.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.babylonml.backend.training.operations.Operation
import com.babylonml.backend.training.operations.Variable

object MatrixEngine {
    /**
     * Convert matrix expression into a backend operation graph and run the forward propagation.
     */
    fun compute(matrix: Matrix): FloatArray {
        val context = TrainingExecutionContext()
        val terminal = toOperationGraph(matrix, context)
        context.initializeExecution(terminal)
        val result = context.executeForwardPropagation()

        val buffer = context.getMemoryBuffer(result)

        val resultOffset = TrainingExecutionContext.addressOffset(result)
        val length = TrainingExecutionContext.addressLength(result)

        return buffer.copyOfRange(resultOffset, resultOffset + length)
    }

    fun toOperationGraph(
        matrix: Matrix,
        context: TrainingExecutionContext
    ): Operation =
        when (matrix) {
            is EagerMatrix -> {
                Variable(
                    context,
                    SimpleGradientDescentOptimizer(1),
                    matrix.data,
                    matrix.dims.rows,
                    matrix.dims.columns,
                    0.001f
                )
            }

            is MatrixExpression -> {
                val left = toOperationGraph(matrix.left, context)
                fun right() = toOperationGraph(matrix.right, context)

                when (matrix.op.type) {
                    MatrixOperationType.Unary ->
                        matrix.op.toBackendOperation(context, left, left, matrix.left.dims, matrix.left.dims)

                    MatrixOperationType.Binary ->
                        matrix.op.toBackendOperation(context, left, right(), matrix.left.dims, matrix.right.dims)
                }
            }
        }
}
