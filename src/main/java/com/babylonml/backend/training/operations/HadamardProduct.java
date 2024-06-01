package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.tornadoml.cpu.VectorOperations;

@SuppressWarnings("unused")
public final class HadamardProduct extends AbstractOperation {
    private final int rows;
    private final int columns;

    private long leftOperationValue;
    private long rightOperationValue;

    private final boolean requiresDerivativeChainValue;

    public HadamardProduct(int rows, int columns, TrainingExecutionContext executionContext,
                           Operation leftOperation, Operation rightOperation) {
        super(executionContext, leftOperation, rightOperation);
        this.rows = rows;
        this.columns = columns;

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue() ||
                rightOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        leftOperationValue = leftOperation.forwardPassCalculation();
        rightOperationValue = rightOperation.forwardPassCalculation();

        var rightOperationValueOffset = TrainingExecutionContext.addressOffset(rightOperationValue);
        var leftOperationValueOffset = TrainingExecutionContext.addressOffset(leftOperationValue);
        assert rows * columns == TrainingExecutionContext.addressLength(leftOperationValue);

        var rightOperationValueBuffer = executionContext.getMemoryBuffer(rightOperationValue);
        var leftOperationValueBuffer = executionContext.getMemoryBuffer(leftOperationValue);
        assert rows * columns == TrainingExecutionContext.addressLength(rightOperationValue);


        var result = executionContext.allocateForwardMemory(rows * columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        VectorOperations.vectorToVectorElementWiseMultiplication(leftOperationValueBuffer,
                leftOperationValueOffset, rightOperationValueBuffer,
                rightOperationValueOffset, resultBuffer, resultOffset, rows * columns);

        return result;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        var rightOperationValueBuffer = executionContext.getMemoryBuffer(rightOperationValue);
        var rightOperationValueOffset = TrainingExecutionContext.addressOffset(rightOperationValue);

        var derivativeChainBuffer = executionContext.getMemoryBuffer(derivativeChainValue);
        var derivativeChainOffset = TrainingExecutionContext.addressOffset(derivativeChainValue);

        var result = executionContext.allocateBackwardMemory(rows * columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);

        VectorOperations.vectorToVectorElementWiseMultiplication(derivativeChainBuffer, derivativeChainOffset,
                rightOperationValueBuffer, rightOperationValueOffset, resultBuffer, 0, rows * columns);

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        var leftOperationValueBuffer = executionContext.getMemoryBuffer(leftOperationValue);
        var leftOperationValueOffset = TrainingExecutionContext.addressOffset(leftOperationValue);

        var derivativeChainBuffer = executionContext.getMemoryBuffer(derivativeChainValue);
        var derivativeChainOffset = TrainingExecutionContext.addressOffset(derivativeChainValue);

        var result = executionContext.allocateBackwardMemory(rows * columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);

        VectorOperations.vectorToVectorElementWiseMultiplication(derivativeChainBuffer, derivativeChainOffset,
                leftOperationValueBuffer, leftOperationValueOffset, resultBuffer, 0, rows * columns);

        return result;
    }

    @Override
    public int getForwardMemorySize() {
        return rows * columns;
    }

    @Override
    public int getBackwardMemorySize() {
        return 2 * rows * columns;
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }
}
