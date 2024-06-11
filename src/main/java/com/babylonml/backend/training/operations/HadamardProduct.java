package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.tornadoml.cpu.VectorOperations;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;

public final class HadamardProduct extends AbstractOperation {

    private long leftOperationPointer;
    private long rightOperationPointer;

    private final boolean requiresDerivativeChainValue;

    private final int maxRows;
    private final int maxColumns;

    public HadamardProduct(TrainingExecutionContext executionContext,
                           Operation leftOperation, Operation rightOperation) {
        super(executionContext, leftOperation, rightOperation);
        this.maxRows = Math.max(leftOperation.getResultMaxRows(), rightOperation.getResultMaxRows());
        this.maxColumns = Math.max(leftOperation.getResultMaxColumns(), rightOperation.getResultMaxColumns());

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue() ||
                rightOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        leftOperationPointer = leftOperation.forwardPassCalculation();
        rightOperationPointer = rightOperation.forwardPassCalculation();

        var leftOperationValueOffset = TrainingExecutionContext.addressOffset(leftOperationPointer);
        var leftOperationValueBuffer = executionContext.getMemoryBuffer(leftOperationPointer);
        var leftOperationRows = TrainingExecutionContext.rows(leftOperationValueBuffer, leftOperationValueOffset);
        var leftOperationColumns = TrainingExecutionContext.columns(leftOperationValueBuffer, leftOperationValueOffset);

        var rightOperationValueOffset = TrainingExecutionContext.addressOffset(rightOperationPointer);
        var rightOperationValueBuffer = executionContext.getMemoryBuffer(rightOperationPointer);
        var rightOperationRows = TrainingExecutionContext.rows(rightOperationValueBuffer, rightOperationValueOffset);
        var rightOperationColumns = TrainingExecutionContext.columns(rightOperationValueBuffer, rightOperationValueOffset);

        assert leftOperationRows == rightOperationRows;
        assert leftOperationColumns == rightOperationColumns;

        assert maxRows >= leftOperationRows;
        assert maxColumns >= leftOperationColumns;

        var result = executionContext.allocateForwardMemory(leftOperationRows, rightOperationColumns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        VectorOperations.vectorToVectorElementWiseMultiplication(leftOperationValueBuffer,
                leftOperationValueOffset, rightOperationValueBuffer,
                rightOperationValueOffset, resultBuffer, resultOffset, leftOperationRows * leftOperationColumns);

        return result;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        var rightOperationValueBuffer = executionContext.getMemoryBuffer(rightOperationPointer);
        var rightOperationValueOffset = TrainingExecutionContext.addressOffset(rightOperationPointer);

        var rightOperationRows = TrainingExecutionContext.rows(rightOperationValueBuffer, rightOperationValueOffset);
        var rightOperationColumns = TrainingExecutionContext.columns(rightOperationValueBuffer, rightOperationValueOffset);

        var derivativeChainBuffer = executionContext.getMemoryBuffer(derivativeChainPointer);
        var derivativeChainOffset = TrainingExecutionContext.addressOffset(derivativeChainPointer);

        var derivativeChainRows = TrainingExecutionContext.rows(derivativeChainBuffer, derivativeChainOffset);
        var derivativeChainColumns = TrainingExecutionContext.columns(derivativeChainBuffer, derivativeChainOffset);

        assert rightOperationRows == derivativeChainRows;
        assert rightOperationColumns == derivativeChainColumns;

        assert maxRows >= rightOperationRows;
        assert maxColumns >= rightOperationColumns;

        var result = executionContext.allocateBackwardMemory(derivativeChainRows, derivativeChainColumns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        VectorOperations.vectorToVectorElementWiseMultiplication(derivativeChainBuffer, derivativeChainOffset,
                rightOperationValueBuffer, rightOperationValueOffset, resultBuffer, resultOffset,
                derivativeChainRows * derivativeChainColumns);

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        var leftOperationValueBuffer = executionContext.getMemoryBuffer(leftOperationPointer);
        var leftOperationValueOffset = TrainingExecutionContext.addressOffset(leftOperationPointer);
        var leftOperationRows = TrainingExecutionContext.rows(leftOperationValueBuffer, leftOperationValueOffset);
        var leftOperationColumns = TrainingExecutionContext.columns(leftOperationValueBuffer, leftOperationValueOffset);

        var derivativeChainBuffer = executionContext.getMemoryBuffer(derivativeChainPointer);
        var derivativeChainOffset = TrainingExecutionContext.addressOffset(derivativeChainPointer);
        var derivativeChainRows = TrainingExecutionContext.rows(derivativeChainBuffer, derivativeChainOffset);
        var derivativeChainColumns = TrainingExecutionContext.columns(derivativeChainBuffer, derivativeChainOffset);

        assert leftOperationRows == derivativeChainRows;
        assert leftOperationColumns == derivativeChainColumns;

        assert maxRows >= leftOperationRows;
        assert maxColumns >= leftOperationColumns;

        var result = executionContext.allocateBackwardMemory(derivativeChainRows, derivativeChainColumns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        VectorOperations.vectorToVectorElementWiseMultiplication(derivativeChainBuffer, derivativeChainOffset,
                leftOperationValueBuffer, leftOperationValueOffset, resultBuffer, resultOffset,
                derivativeChainRows * derivativeChainColumns);

        return result;
    }

    @Override
    public IntIntImmutablePair[] getForwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(maxRows, maxColumns)
        };
    }

    @Override
    public IntIntImmutablePair[] getBackwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(maxRows, maxColumns),
                new IntIntImmutablePair(maxRows, maxColumns),
        };
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }

    @Override
    public int getResultMaxRows() {
        return maxRows;
    }

    @Override
    public int getResultMaxColumns() {
        return maxColumns;
    }
}
