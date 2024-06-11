package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class MSEByRowsCostFunction extends AbstractOperation {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private final int maxRows;
    private final int maxColumns;

    private final boolean requiresDerivativeChainValue;

    private long predictionOperandPointer;
    private long expectedValuesPointer;

    public MSEByRowsCostFunction(TrainingExecutionContext executionContext, Operation predictionOperation,
                                 Operation expectedValuesOperation) {
        this(null, executionContext, predictionOperation, expectedValuesOperation);
    }

    public MSEByRowsCostFunction(String name,
                                 TrainingExecutionContext executionContext, Operation predictionOperation,
                                 Operation expectedValuesOperation) {
        super(name, executionContext, predictionOperation, expectedValuesOperation);
        this.maxRows = leftOperation.getResultMaxRows();
        this.maxColumns = leftOperation.getResultMaxColumns();

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        predictionOperandPointer = leftOperation.forwardPassCalculation();

        var predictionOperandBuffer = executionContext.getMemoryBuffer(predictionOperandPointer);
        var predictionOperandOffset = TrainingExecutionContext.addressOffset(predictionOperandPointer);
        var predictionOperandRows = TrainingExecutionContext.rows(predictionOperandBuffer, predictionOperandOffset);
        var predictionOperandColumns = TrainingExecutionContext.columns(predictionOperandBuffer, predictionOperandOffset);

        expectedValuesPointer = rightOperation.forwardPassCalculation();
        var expectedValuesBuffer = executionContext.getMemoryBuffer(expectedValuesPointer);
        var expectedValuesOffset = TrainingExecutionContext.addressOffset(expectedValuesPointer);

        var result = executionContext.allocateForwardMemory(1, 1);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        var loopBound = SPECIES.loopBound(predictionOperandRows * predictionOperandColumns);
        var vecSum = FloatVector.zero(SPECIES);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var vec = FloatVector.fromArray(SPECIES, predictionOperandBuffer, predictionOperandOffset + i);
            var expectedVec = FloatVector.fromArray(SPECIES, expectedValuesBuffer, i + expectedValuesOffset);

            var diff = vec.sub(expectedVec);
            vecSum = diff.fma(diff, vecSum);
        }

        var sum = vecSum.reduceLanes(VectorOperators.ADD);
        for (int i = loopBound; i < predictionOperandRows * predictionOperandColumns; i++) {
            var value = predictionOperandBuffer[predictionOperandOffset + i] -
                    expectedValuesBuffer[i + expectedValuesOffset];
            sum += value * value;
        }

        resultBuffer[resultOffset] = sum / predictionOperandRows;

        return result;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        var predictionOperandBuffer = executionContext.getMemoryBuffer(predictionOperandPointer);
        var predictionOperandOffset = TrainingExecutionContext.addressOffset(predictionOperandPointer);
        var predictionOperandRows = TrainingExecutionContext.rows(predictionOperandBuffer, predictionOperandOffset);
        var predictionOperandColumns = TrainingExecutionContext.columns(predictionOperandBuffer, predictionOperandOffset);

        var expectedValues = executionContext.getMemoryBuffer(expectedValuesPointer);
        var expectedValuesOffset = TrainingExecutionContext.addressOffset(expectedValuesPointer);

        var result = executionContext.allocateBackwardMemory(predictionOperandRows, predictionOperandColumns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        var loopBound = SPECIES.loopBound(predictionOperandRows * predictionOperandColumns);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var vec = FloatVector.fromArray(SPECIES, predictionOperandBuffer, predictionOperandOffset + i);
            var expectedVec = FloatVector.fromArray(SPECIES, expectedValues, i + expectedValuesOffset);

            var diff = vec.sub(expectedVec);
            diff.intoArray(resultBuffer, resultOffset + i);
        }

        for (int i = loopBound; i < predictionOperandRows * predictionOperandColumns; i++) {
            resultBuffer[i + resultOffset] = predictionOperandBuffer[i + predictionOperandOffset]
                    - expectedValues[i + expectedValuesOffset];
        }

        return result;
    }

    @Override
    public int getResultMaxRows() {
        return 1;
    }

    @Override
    public int getResultMaxColumns() {
        return 1;
    }

    @Override
    public IntIntImmutablePair[] getForwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(1, 1)
        };
    }

    @Override
    public IntIntImmutablePair[] getBackwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(maxRows, maxColumns)
        };
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }
}
