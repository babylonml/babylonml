package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.tornadoml.cpu.VectorOperations;


public final class Add extends AbstractOperation {
    private final int rows;
    private final int columns;

    private long derivativeResult;

    private final boolean requiresDerivativeChainValue;

    public Add(TrainingExecutionContext executionContext, int rows, int columns, Operation leftOperation,
               Operation rightOperation) {
        super(executionContext, leftOperation, rightOperation);

        this.rows = rows;
        this.columns = columns;

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue()
                || rightOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        var leftResult = leftOperation.forwardPassCalculation();
        var rightValue = rightOperation.forwardPassCalculation();

        assert TrainingExecutionContext.addressLength(leftResult) == rows * columns;
        assert TrainingExecutionContext.addressLength(rightValue) == rows * columns;

        var result = executionContext.allocateForwardMemory(rows * columns);
        var resultOffset = TrainingExecutionContext.addressOffset(result);
        var resultBuffer = executionContext.getMemoryBuffer(result);

        var leftBuffer = executionContext.getMemoryBuffer(leftResult);
        var leftOffset = TrainingExecutionContext.addressOffset(leftResult);

        var rightBuffer = executionContext.getMemoryBuffer(rightValue);
        var rightOffset = TrainingExecutionContext.addressOffset(rightValue);

        VectorOperations.addVectorToVector(leftBuffer, leftOffset, rightBuffer, rightOffset,
                resultBuffer, resultOffset, rows * columns);

        return result;
    }

    @Override
    public int getForwardMemorySize() {
        return rows * columns;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        return calculateDerivative();
    }

    private long calculateDerivative() {
        if (TrainingExecutionContext.isNull(derivativeResult)) {
            derivativeResult = executionContext.allocateBackwardMemory(rows * columns);
            var resultBuffer = executionContext.getMemoryBuffer(derivativeResult);
            var resultOffset = TrainingExecutionContext.addressOffset(derivativeResult);

            var derivativeChainBuffer = executionContext.getMemoryBuffer(derivativeChainValue);
            var derivativeChainValueOffset = TrainingExecutionContext.addressOffset(derivativeChainValue);

            System.arraycopy(derivativeChainBuffer, derivativeChainValueOffset, resultBuffer,
                    resultOffset, rows * columns);
        }

        return derivativeResult;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return calculateDerivative();
    }


    @Override
    public int getBackwardMemorySize() {
        return rows * columns;
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }

    @Override
    public void reset() {
        super.reset();

        derivativeResult = TrainingExecutionContext.NULL;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }
}
