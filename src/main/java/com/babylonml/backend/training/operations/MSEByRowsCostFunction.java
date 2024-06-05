package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class MSEByRowsCostFunction extends AbstractOperation {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private final float[] expectedValues;

    private final int rows;
    private final int columns;

    private final boolean requiresDerivativeChainValue;

    private long leftOperationResult;

    public MSEByRowsCostFunction(TrainingExecutionContext executionContext, Operation leftOperation,
                                 int rows, int columns, float[] expectedValues) {
        super(executionContext, leftOperation, null);
        this.rows = rows;
        this.columns = columns;

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue();
        this.expectedValues = expectedValues;
    }

    @Override
    public long forwardPassCalculation() {
        leftOperationResult = leftOperation.forwardPassCalculation();

        var leftBuffer = executionContext.getMemoryBuffer(leftOperationResult);
        var leftOffset = TrainingExecutionContext.addressOffset(leftOperationResult);

        var result = executionContext.allocateForwardMemory(1);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        var loopBound = SPECIES.loopBound(rows * columns);
        var vecSum = FloatVector.zero(SPECIES);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var vec = FloatVector.fromArray(SPECIES, leftBuffer, leftOffset + i);
            var expectedVec = FloatVector.fromArray(SPECIES, expectedValues, i);

            var diff = vec.sub(expectedVec);
            vecSum = diff.fma(diff, vecSum);
        }

        var sum = vecSum.reduceLanes(VectorOperators.ADD);
        for (int i = loopBound; i < rows * columns; i++) {
            var value = leftBuffer[leftOffset + i] - expectedValues[i];
            sum += value * value;
        }

        resultBuffer[resultOffset] = sum / rows;

        return result;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        var leftBuffer = executionContext.getMemoryBuffer(leftOperationResult);
        var leftOffset = TrainingExecutionContext.addressOffset(leftOperationResult);

        var result = executionContext.allocateBackwardMemory(rows * columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        var loopBound = SPECIES.loopBound(rows * columns);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var vec = FloatVector.fromArray(SPECIES, leftBuffer, leftOffset + i);
            var expectedVec = FloatVector.fromArray(SPECIES, expectedValues, i);

            var diff = vec.sub(expectedVec);
            diff.intoArray(resultBuffer, resultOffset + i);
        }

        for (int i = loopBound; i < rows * columns; i++) {
            resultBuffer[i + resultOffset] = leftBuffer[i + leftOffset] - expectedValues[i];
        }

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public int getForwardMemorySize() {
        return 1;
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
