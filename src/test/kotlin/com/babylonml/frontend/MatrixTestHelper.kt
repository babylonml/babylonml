package com.babylonml.frontend

import com.babylonml.backend.training.TrainingExecutionContext
import com.babylonml.backend.training.operations.AbstractOperation
import com.babylonml.backend.training.operations.Operation
import com.babylonml.backend.training.operations.Variable
import org.junit.jupiter.api.Assertions.*
import kotlin.reflect.KClass
import kotlin.reflect.cast

object MatrixTestHelper {

    fun toOpGraph(matrix: Matrix): Operation =
        MatrixEngine.toOperationGraph(matrix, TrainingExecutionContext())

    fun <Op : AbstractOperation> validateOperation(
        operationClass: KClass<Op>,
        result: Operation?,
        check: (Op) -> Unit = { },
        left: (Operation?) -> Unit = { },
        right: (Operation?) -> Unit = { }
    ) {
        assertInstanceOf(operationClass.java, result)

        val operation = operationClass.cast(result)

        check(operation)
        left(operation.leftPreviousOperation)
        right(operation.rightPreviousOperation)
    }

    fun validateVariable(operation: Operation?, input: EagerMatrix) {
        assertInstanceOf(Variable::class.java, operation)

        val variable = operation as Variable

        assertEquals(input.dims.rows, variable.rows)
        assertEquals(input.dims.columns, variable.columns)
        assertArrayEquals(input.data, variable.data)
    }
}
