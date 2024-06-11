package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.tornadoml.cpu.MatrixOperations;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;

public final class Multiplication extends AbstractOperation {
    private final int leftMatrixMaxRows;
    private final int leftMatrixMaxColumns;

    private final int rightMatrixMaxRows;
    private final int rightMatrixMaxColumns;

    private long leftOperandResultPointer;
    private long rightOperandResultPointer;

    private final boolean requiresDerivativeChainValue;

    public Multiplication(TrainingExecutionContext executionContext,
                          Operation leftOperation, Operation rightOperation) {
        this(null, executionContext, leftOperation, rightOperation);
    }

    public Multiplication(String name, TrainingExecutionContext executionContext,
                          Operation leftOperation, Operation rightOperation) {
        super(name, executionContext, leftOperation, rightOperation);

        this.leftMatrixMaxRows = leftOperation.getResultMaxRows();
        this.leftMatrixMaxColumns = leftOperation.getResultMaxColumns();

        this.rightMatrixMaxRows = rightOperation.getResultMaxRows();
        this.rightMatrixMaxColumns = rightOperation.getResultMaxColumns();

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue()
                || rightOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        leftOperandResultPointer = leftOperation.forwardPassCalculation();
        var leftOperandBuffer = executionContext.getMemoryBuffer(leftOperandResultPointer);
        var leftOperandOffset = TrainingExecutionContext.addressOffset(leftOperandResultPointer);

        var leftOperandRows = TrainingExecutionContext.rows(leftOperandBuffer, leftOperandOffset);
        var leftOperandColumns = TrainingExecutionContext.columns(leftOperandBuffer, leftOperandOffset);

        rightOperandResultPointer = rightOperation.forwardPassCalculation();
        var rightOperandBuffer = executionContext.getMemoryBuffer(rightOperandResultPointer);
        var rightOperandOffset = TrainingExecutionContext.addressOffset(rightOperandResultPointer);

        var rightOperationRows = TrainingExecutionContext.rows(rightOperandBuffer, rightOperandOffset);
        var rightOperationColumns = TrainingExecutionContext.columns(rightOperandBuffer, rightOperandOffset);

        assert leftOperandColumns == rightOperationRows;


        var result = executionContext.allocateForwardMemory(leftOperandRows, rightOperationColumns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        MatrixOperations.matrixToMatrixMultiplication(leftOperandBuffer, leftOperandOffset, leftMatrixMaxRows, leftMatrixMaxColumns,
                rightOperandBuffer, rightOperandOffset,
                leftMatrixMaxColumns, rightMatrixMaxColumns, resultBuffer, resultOffset);

        return result;
    }

    @Override
    public IntIntImmutablePair[] getForwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(leftMatrixMaxRows, rightMatrixMaxColumns)
        };
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        var rightOperandBuffer = executionContext.getMemoryBuffer(rightOperandResultPointer);
        var rightOperandOffset = TrainingExecutionContext.addressOffset(rightOperandResultPointer);

        var rightOperandRows = TrainingExecutionContext.rows(rightOperandBuffer, rightOperandOffset);
        var rightOperandColumns = TrainingExecutionContext.columns(rightOperandBuffer, rightOperandOffset);

        var derivativeBuffer = executionContext.getMemoryBuffer(derivativeChainPointer);
        var derivativeOffset = TrainingExecutionContext.addressOffset(derivativeChainPointer);
        var derivativeRows = TrainingExecutionContext.rows(derivativeBuffer, derivativeOffset);
        var derivativeColumns = TrainingExecutionContext.columns(derivativeBuffer, derivativeOffset);

        //right^T
        var rightTranspose = executionContext.allocateBackwardMemory(rightOperandColumns, rightOperandRows);
        var rightTransposeOffset = TrainingExecutionContext.addressOffset(rightTranspose);
        var rightTransposeBuffer = executionContext.getMemoryBuffer(rightTranspose);

        MatrixOperations.transposeMatrix(rightOperandBuffer, rightOperandOffset, rightOperandRows, rightOperandColumns,
                rightTransposeBuffer, rightTransposeOffset);

        var result = executionContext.allocateBackwardMemory(derivativeRows, rightOperandRows);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        //leftDerivative = derivative * right^T
        MatrixOperations.matrixToMatrixMultiplication(derivativeBuffer, derivativeOffset, derivativeRows, derivativeColumns,
                rightTransposeBuffer, rightTransposeOffset, rightOperandColumns, rightOperandRows,
                resultBuffer, resultOffset);

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        var leftOperandBuffer = executionContext.getMemoryBuffer(leftOperandResultPointer);
        var leftOperandOffset = TrainingExecutionContext.addressOffset(leftOperandResultPointer);

        var leftOperandRows = TrainingExecutionContext.rows(leftOperandBuffer, leftOperandOffset);
        var leftOperandColumns = TrainingExecutionContext.columns(leftOperandBuffer, leftOperandOffset);

        //left^T
        var leftTranspose = executionContext.allocateBackwardMemory(leftOperandColumns, leftOperandRows);
        var leftTransposeBuffer = executionContext.getMemoryBuffer(leftTranspose);
        var leftTransposeOffset = TrainingExecutionContext.addressOffset(leftTranspose);

        MatrixOperations.transposeMatrix(leftOperandBuffer, leftOperandOffset, leftOperandRows, leftOperandColumns,
                leftTransposeBuffer, leftTransposeOffset);

        var derivativeBuffer = executionContext.getMemoryBuffer(derivativeChainPointer);
        var derivativeOffset = TrainingExecutionContext.addressOffset(derivativeChainPointer);

        var derivativeRows = TrainingExecutionContext.rows(derivativeBuffer, derivativeOffset);
        var derivativeColumns = TrainingExecutionContext.columns(derivativeBuffer, derivativeOffset);

        var result = executionContext.allocateBackwardMemory(leftOperandColumns, derivativeColumns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        //rightDerivative = left^T * derivative
        MatrixOperations.matrixToMatrixMultiplication(leftTransposeBuffer, leftTransposeOffset, leftOperandColumns, leftOperandRows,
                derivativeBuffer, derivativeOffset, derivativeRows, derivativeColumns, resultBuffer, resultOffset);

        return result;
    }

    @Override
    public IntIntImmutablePair[] getBackwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                //left
                new IntIntImmutablePair(leftMatrixMaxRows, leftMatrixMaxColumns),
                //right^t
                new IntIntImmutablePair(rightMatrixMaxColumns, rightMatrixMaxRows),


                //right
                new IntIntImmutablePair(rightMatrixMaxRows, rightMatrixMaxColumns),
                //left^t
                new IntIntImmutablePair(leftMatrixMaxColumns, leftMatrixMaxRows),
        };
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }

    @Override
    public int getResultMaxRows() {
        return leftMatrixMaxRows;
    }

    @Override
    public int getResultMaxColumns() {
        return rightMatrixMaxColumns;
    }
}
