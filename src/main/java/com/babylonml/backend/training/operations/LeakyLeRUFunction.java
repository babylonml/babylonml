package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class LeakyLeRUFunction extends AbstractOperation {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private final int maxRows;
    private final int maxColumns;

    private final boolean requiresDerivativeChainValue;
    private final float leakyLeRUASlope;

    private long leftResultPointer;

    public LeakyLeRUFunction(float leakyLeRUASlope, TrainingExecutionContext executionContext,
                             Operation leftOperation) {
        this(null, leakyLeRUASlope, executionContext, leftOperation);
    }

    public LeakyLeRUFunction(String name, float leakyLeRUASlope, TrainingExecutionContext executionContext,
                             Operation leftOperation) {
        super(name, executionContext, leftOperation, null);
        this.leakyLeRUASlope = leakyLeRUASlope;

        this.maxRows = leftOperation.getResultMaxRows();
        this.maxColumns = leftOperation.getResultMaxColumns();

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        leftResultPointer = leftOperation.forwardPassCalculation();

        var leftResultBuffer = executionContext.getMemoryBuffer(leftResultPointer);
        var leftResultOffset = TrainingExecutionContext.addressOffset(leftResultPointer);

        var rows = TrainingExecutionContext.rows(leftResultBuffer, leftResultOffset);
        var columns = TrainingExecutionContext.columns(leftResultBuffer, leftResultOffset);

        assert maxColumns >= columns;
        assert maxRows >= rows;

        var result = executionContext.allocateForwardMemory(rows, columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        var size = rows * columns;
        var loopBound = SPECIES.loopBound(size);
        var zero = FloatVector.zero(SPECIES);

        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, leftResultBuffer, leftResultOffset + i);
            var mask = va.compare(VectorOperators.LT, zero);
            var vc = va.mul(leakyLeRUASlope, mask);
            vc.intoArray(resultBuffer, resultOffset + i);
        }

        for (int i = loopBound; i < size; i++) {
            var leftValue = leftResultBuffer[leftResultOffset + i];
            resultBuffer[i + resultOffset] = leftValue > 0 ? leftValue : leakyLeRUASlope * leftValue;
        }

        return result;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        var leftResultBuffer = executionContext.getMemoryBuffer(leftResultPointer);
        var leftResultOffset = TrainingExecutionContext.addressOffset(leftResultPointer);
        var leftRows = TrainingExecutionContext.rows(leftResultBuffer, leftResultOffset);
        var leftColumns = TrainingExecutionContext.columns(leftResultBuffer, leftResultOffset);

        var derivativeChainBuffer = executionContext.getMemoryBuffer(derivativeChainPointer);
        var derivativeChainOffset = TrainingExecutionContext.addressOffset(derivativeChainPointer);
        var derivativeRows = TrainingExecutionContext.rows(derivativeChainBuffer, derivativeChainOffset);
        var derivativeColumns = TrainingExecutionContext.columns(derivativeChainBuffer, derivativeChainOffset);

        assert leftRows == derivativeRows;
        assert leftColumns == derivativeColumns;

        assert maxRows >= leftRows;
        assert maxColumns >= leftColumns;

        var result = executionContext.allocateBackwardMemory(leftRows, leftColumns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        var size = leftRows * leftColumns;
        var loopBound = SPECIES.loopBound(size);
        var zero = FloatVector.zero(SPECIES);
        var slope = FloatVector.broadcast(SPECIES, leakyLeRUASlope);
        var one = FloatVector.broadcast(SPECIES, 1.0f);

        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, leftResultBuffer, leftResultOffset + i);
            var mask = va.compare(VectorOperators.LT, zero);
            var vc = one.mul(slope, mask);

            var diff = FloatVector.fromArray(SPECIES, derivativeChainBuffer, derivativeChainOffset + i);
            vc = vc.mul(diff);

            vc.intoArray(resultBuffer, resultOffset + i);
        }

        for (int i = loopBound; i < size; i++) {
            resultBuffer[i + resultOffset] = (leftResultBuffer[i + leftResultOffset] > 0 ? 1.0f : leakyLeRUASlope) *
                    derivativeChainBuffer[i + derivativeChainOffset];
        }

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
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

    @Override
    public IntIntImmutablePair[] getForwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(maxRows, maxColumns)
        };
    }

    @Override
    public IntIntImmutablePair[] getBackwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(maxRows, maxColumns)
        };
    }
}
