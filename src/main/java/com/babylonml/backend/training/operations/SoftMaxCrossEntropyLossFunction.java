package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.tornadoml.cpu.VectorOperations;

@SuppressWarnings("unused")
public final class SoftMaxCrossEntropyLossFunction extends AbstractOperation {
    private final int rows;
    private final int columns;

    private long leftOperationResult;
    private final float[] expectedValue;

    private final boolean requiresDerivativeChainValue;

    public SoftMaxCrossEntropyLossFunction(int rows,  int columns, float[] expectedValue, TrainingExecutionContext executionContext,
                                           Operation leftOperation) {
        super(executionContext, leftOperation, null);

        this.rows = rows;
        this.columns = columns;
        this.expectedValue = expectedValue;

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        leftOperationResult = leftOperation.forwardPassCalculation();

        //we do nothing with result.
        return TrainingExecutionContext.NULL;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        var leftBuffer = executionContext.getMemoryBuffer(leftOperationResult);
        var leftOffset = TrainingExecutionContext.addressOffset(leftOperationResult);

        var result = executionContext.allocateBackwardMemory(rows * columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        VectorOperations.subtractVectorFromVector(leftBuffer, leftOffset, expectedValue, 0,
                resultBuffer, resultOffset, rows * columns);

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public int getForwardMemorySize() {
        return 0;
    }

    @Override
    public int getBackwardMemorySize() {
        return rows * columns;
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }
}
