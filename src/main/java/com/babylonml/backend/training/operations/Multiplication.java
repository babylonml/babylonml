package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.tornadoml.cpu.MatrixOperations;

public final class Multiplication extends AbstractOperation {
    private final int firstMatrixRows;
    private final int firstMatrixColumns;

    private final int secondMatrixColumns;

    private long leftOperationResult;
    private long rightOperationResult;

    private final boolean requiresDerivativeChainValue;

    public Multiplication(TrainingExecutionContext executionContext, int firstMatrixRows, int firstMatrixColumns,
                          int secondMatrixColumns, Operation leftOperation, Operation rightOperation) {
        super(executionContext, leftOperation, rightOperation);

        this.firstMatrixRows = firstMatrixRows;
        this.firstMatrixColumns = firstMatrixColumns;
        this.secondMatrixColumns = secondMatrixColumns;
        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue()
                || rightOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        leftOperationResult = leftOperation.forwardPassCalculation();
        rightOperationResult = rightOperation.forwardPassCalculation();

        assert TrainingExecutionContext.addressLength(leftOperationResult) == firstMatrixRows * firstMatrixColumns;
        assert TrainingExecutionContext.addressLength(rightOperationResult) == firstMatrixColumns * secondMatrixColumns;

        var result = executionContext.allocateForwardMemory(firstMatrixRows * secondMatrixColumns);

        var leftBuffer = executionContext.getMemoryBuffer(leftOperationResult);
        var rightBuffer = executionContext.getMemoryBuffer(rightOperationResult);
        var resultBuffer = executionContext.getMemoryBuffer(result);

        var leftMatrixOffset = TrainingExecutionContext.addressOffset(leftOperationResult);
        var rightMatrixOffset = TrainingExecutionContext.addressOffset(rightOperationResult);

        var resultOffset = TrainingExecutionContext.addressOffset(result);

        MatrixOperations.matrixToMatrixMultiplication(leftBuffer, leftMatrixOffset, firstMatrixRows, firstMatrixColumns,
                rightBuffer, rightMatrixOffset, firstMatrixColumns, secondMatrixColumns, resultBuffer, resultOffset);

        return result;
    }

    @Override
    public int getForwardMemorySize() {
        return firstMatrixRows * secondMatrixColumns;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {

        //right^T
        var rightTranspose = executionContext.allocateBackwardMemory(firstMatrixColumns * secondMatrixColumns);
        var rightTransposeOffset = TrainingExecutionContext.addressOffset(rightTranspose);
        var rightTransposeBuffer = executionContext.getMemoryBuffer(rightTranspose);

        var rightValueBuffer = executionContext.getMemoryBuffer(rightOperationResult);
        var rightValueOffset = TrainingExecutionContext.addressOffset(rightOperationResult);

        MatrixOperations.transposeMatrix(rightValueBuffer, rightValueOffset, firstMatrixColumns, secondMatrixColumns,
                rightTransposeBuffer, rightTransposeOffset);

        var derivativeBuffer = executionContext.getMemoryBuffer(derivativeChainValue);
        var derivativeOffset = TrainingExecutionContext.addressOffset(derivativeChainValue);
        assert TrainingExecutionContext.addressLength(derivativeChainValue) == firstMatrixRows * secondMatrixColumns;

        var result = executionContext.allocateBackwardMemory(firstMatrixRows * firstMatrixColumns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        //leftDerivative = derivative * right^T
        MatrixOperations.matrixToMatrixMultiplication(derivativeBuffer, derivativeOffset, firstMatrixRows, secondMatrixColumns,
                rightTransposeBuffer, rightTransposeOffset, secondMatrixColumns, firstMatrixColumns, resultBuffer, resultOffset);

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        var leftResultBuffer = executionContext.getMemoryBuffer(leftOperationResult);
        var leftOffset = TrainingExecutionContext.addressOffset(leftOperationResult);
        assert TrainingExecutionContext.addressLength(leftOperationResult) == firstMatrixRows * firstMatrixColumns;

        //left^T
        var leftTranspose = executionContext.allocateBackwardMemory(firstMatrixRows * firstMatrixColumns);
        var leftTransposeBuffer = executionContext.getMemoryBuffer(leftTranspose);
        var leftTransposeOffset = TrainingExecutionContext.addressOffset(leftTranspose);

        MatrixOperations.transposeMatrix(leftResultBuffer, leftOffset, firstMatrixRows, firstMatrixColumns,
                leftTransposeBuffer, leftTransposeOffset);

        var derivativeBuffer = executionContext.getMemoryBuffer(derivativeChainValue);
        var derivativeOffset = TrainingExecutionContext.addressOffset(derivativeChainValue);
        assert TrainingExecutionContext.addressLength(derivativeChainValue) == firstMatrixRows * secondMatrixColumns;


        var result = executionContext.allocateBackwardMemory(firstMatrixColumns * secondMatrixColumns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        //rightDerivative = left^T * derivative
        MatrixOperations.matrixToMatrixMultiplication(leftTransposeBuffer, leftTransposeOffset, firstMatrixColumns, firstMatrixRows,
                derivativeBuffer, derivativeOffset, firstMatrixRows, secondMatrixColumns, resultBuffer, resultOffset);

        return result;
    }


    @Override
    public int getBackwardMemorySize() {
        return 2 * (firstMatrixColumns * secondMatrixColumns +
                firstMatrixRows * firstMatrixColumns);
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }
}
