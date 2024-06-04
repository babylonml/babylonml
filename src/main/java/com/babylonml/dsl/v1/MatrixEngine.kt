package com.babylonml.dsl.v1

import com.babylonml.backend.training.SimpleGradientDescentOptimizer
import com.babylonml.backend.training.TrainingExecutionContext
import com.babylonml.backend.training.operations.Operation
import com.babylonml.backend.training.operations.Variable

object MatrixEngine {
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

    private fun toOperationGraph(matrix: Matrix, context: TrainingExecutionContext): Operation =
        when (matrix) {
            is EagerMatrix -> {
                Variable(
                    context,
                    SimpleGradientDescentOptimizer(1),
                    matrix.data,
                    matrix.dims.rows,
                    matrix.dims.cols,
                    0.001f
                )
            }

            is MatrixExpression -> {
                val left = toOperationGraph(matrix.left, context)
                fun right() = toOperationGraph(matrix.right, context)

                when (matrix.op.type) {
                    MatrixOperationType.Unary ->
                        matrix.op.toOperation(context, left, left, matrix.left.dims, matrix.left.dims)

                    MatrixOperationType.Binary ->
                        matrix.op.toOperation(context, left, right(), matrix.left.dims, matrix.right.dims)
                }
            }
        }
}
