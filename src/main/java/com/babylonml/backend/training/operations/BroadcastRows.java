package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.tornadoml.cpu.MatrixOperations;

public final class BroadcastRows extends AbstractOperation {
    private final int columns;
    private final int rows;
    private final boolean requiresDerivativeChainValue;

    public BroadcastRows(final int rows, int columns, TrainingExecutionContext executionContext,
                            Operation leftOperation) {
        super(executionContext, leftOperation, null);
        this.columns = columns;
        this.rows = rows;

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        var leftResult = leftOperation.forwardPassCalculation();
        var leftBuffer = executionContext.getMemoryBuffer(leftResult);
        var leftOffset = TrainingExecutionContext.addressOffset(leftResult);
        assert columns == TrainingExecutionContext.addressLength(leftResult);

        var result = executionContext.allocateForwardMemory(rows * columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        MatrixOperations.broadcastVectorToMatrixByRows(leftBuffer, leftOffset, resultBuffer, resultOffset, rows, columns);

        return result;
    }


    @Override
    public int getForwardMemorySize() {
        return rows * columns;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        assert TrainingExecutionContext.addressLength(derivativeChainValue) == rows * columns;

        var derivativeBuffer = executionContext.getMemoryBuffer(derivativeChainValue);
        var derivativeOffset = TrainingExecutionContext.addressOffset(derivativeChainValue);


        var result = executionContext.allocateBackwardMemory(columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        MatrixOperations.reduceMatrixToVectorByRows(derivativeBuffer, derivativeOffset, rows, columns,
                resultBuffer, resultOffset);

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }


    @Override
    public int getBackwardMemorySize() {
        return columns;
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }
}
