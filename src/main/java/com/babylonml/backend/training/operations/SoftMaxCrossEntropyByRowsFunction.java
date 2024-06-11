package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.tornadoml.cpu.MatrixOperations;
import com.tornadoml.cpu.VectorOperations;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.jspecify.annotations.NonNull;

public final class SoftMaxCrossEntropyByRowsFunction extends AbstractOperation {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private final int maxRows;
    private final int maxColumns;

    private long softMaxResultPointer;
    private long expectedProbabilityPointer;

    private final boolean requiresDerivativeChainValue;

    public SoftMaxCrossEntropyByRowsFunction(@NonNull Operation expectedProbability,
                                             @NonNull TrainingExecutionContext executionContext,
                                             @NonNull Operation predictedOperation) {
        super(executionContext, predictedOperation, expectedProbability);

        this.maxRows = predictedOperation.getResultMaxRows();
        this.maxColumns = predictedOperation.getResultMaxColumns();

        this.requiresDerivativeChainValue = predictedOperation.requiresBackwardDerivativeChainValue();
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
    public long forwardPassCalculation() {
        long predictedOperandResultPointer = leftOperation.forwardPassCalculation();
        var predictedOperandBuffer = executionContext.getMemoryBuffer(predictedOperandResultPointer);
        var predictedOperandOffset = TrainingExecutionContext.addressOffset(predictedOperandResultPointer);

        var predictedOperandRows = TrainingExecutionContext.rows(predictedOperandBuffer, predictedOperandOffset);
        var predictedOperandColumns = TrainingExecutionContext.columns(predictedOperandBuffer, predictedOperandOffset);

        softMaxResultPointer = executionContext.allocateForwardMemory(predictedOperandRows, predictedOperandColumns);

        var softMaxBuffer = executionContext.getMemoryBuffer(softMaxResultPointer);
        var softMaxOffset = TrainingExecutionContext.addressOffset(softMaxResultPointer);

        expectedProbabilityPointer = rightOperation.forwardPassCalculation();
        var expectedProbability = executionContext.getMemoryBuffer(expectedProbabilityPointer);
        var expectedProbabilityOffset = TrainingExecutionContext.addressOffset(expectedProbabilityPointer);

        MatrixOperations.softMaxByRows(predictedOperandBuffer, predictedOperandOffset, predictedOperandRows, predictedOperandColumns,
                softMaxBuffer, softMaxOffset);

        var loopBound = SPECIES.loopBound(maxRows * maxColumns);
        var vecSum = FloatVector.zero(SPECIES);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var vec = FloatVector.fromArray(SPECIES, softMaxBuffer,
                    softMaxOffset + i).lanewise(VectorOperators.LOG);
            var expectedVec = FloatVector.fromArray(SPECIES, expectedProbability, i + expectedProbabilityOffset);
            vecSum = vec.fma(expectedVec, vecSum);
        }

        var sum = vecSum.reduceLanes(VectorOperators.ADD);
        for (int i = loopBound; i < maxRows * maxColumns; i++) {
            sum += (float) Math.log(softMaxBuffer[softMaxOffset + i]) * expectedProbability[i + expectedProbabilityOffset];
        }

        var result = executionContext.allocateForwardMemory(1, 1);

        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);
        resultBuffer[resultOffset] = -sum / maxColumns;

        return result;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        var softMaxBuffer = executionContext.getMemoryBuffer(softMaxResultPointer);
        var softMaxOffset = TrainingExecutionContext.addressOffset(softMaxResultPointer);

        var softMaxRows = TrainingExecutionContext.rows(softMaxBuffer, softMaxOffset);
        var softMaxColumns = TrainingExecutionContext.columns(softMaxBuffer, softMaxOffset);

        var expectedProbability = executionContext.getMemoryBuffer(expectedProbabilityPointer);
        var expectedProbabilityOffset = TrainingExecutionContext.addressOffset(expectedProbabilityPointer);

        var result = executionContext.allocateBackwardMemory(softMaxRows, softMaxColumns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        VectorOperations.subtractVectorFromVector(softMaxBuffer, softMaxOffset, expectedProbability,
                expectedProbabilityOffset, resultBuffer, resultOffset, softMaxRows * softMaxColumns);

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public IntIntImmutablePair[] getForwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(maxRows, maxColumns),
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
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }
}
