package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.tornadoml.cpu.MatrixOperations;
import com.tornadoml.cpu.VectorOperations;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class SoftMaxCrossEntropyByRowsFunction extends AbstractOperation {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private final int rows;
    private final int columns;

    private long leftOperationResult;
    private final float[] expectedProbability;

    private final boolean requiresDerivativeChainValue;

    public SoftMaxCrossEntropyByRowsFunction(int rows, int columns, float[] expectedProbability,
                                             TrainingExecutionContext executionContext,
                                             Operation leftOperation) {
        super(executionContext, leftOperation, null);

        this.rows = rows;
        this.columns = columns;
        this.expectedProbability = expectedProbability;

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        leftOperationResult = leftOperation.forwardPassCalculation();

        var leftOperationBuffer = executionContext.getMemoryBuffer(leftOperationResult);
        var leftOperationOffset = TrainingExecutionContext.addressOffset(leftOperationResult);

        assert rows * columns == TrainingExecutionContext.addressLength(leftOperationResult);

        var calculation = executionContext.allocateForwardMemory(rows * columns);
        var calculationBuffer = executionContext.getMemoryBuffer(calculation);
        var calculationOffset = TrainingExecutionContext.addressOffset(calculation);

        MatrixOperations.softMaxByRows(leftOperationBuffer, leftOperationOffset, rows, columns,
                calculationBuffer, calculationOffset);

        var loopBound = SPECIES.loopBound(rows * columns);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var vec = FloatVector.fromArray(SPECIES, calculationBuffer,
                    calculationOffset + i).lanewise(VectorOperators.LOG);
            vec.intoArray(calculationBuffer, calculationOffset + i);
        }

        var vecSum = FloatVector.zero(SPECIES);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var vec = FloatVector.fromArray(SPECIES, calculationBuffer, calculationOffset + i);
            var expectedVec = FloatVector.fromArray(SPECIES, expectedProbability, i);

            vecSum = vec.fma(expectedVec, vecSum);
        }

        var sum = vecSum.reduceLanes(VectorOperators.ADD);
        for (int i = loopBound; i < rows * columns; i++) {
            calculationBuffer[calculationOffset + i] = (float) Math.log(calculationBuffer[calculationOffset + i]);
        }
        for (int i = loopBound; i < rows * columns; i++) {
            sum += expectedProbability[i] * calculationBuffer[calculationOffset + i];
        }

        var result = executionContext.allocateForwardMemory(1);

        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);
        resultBuffer[resultOffset] = -sum / columns;

        return result;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        var leftBuffer = executionContext.getMemoryBuffer(leftOperationResult);
        var leftOffset = TrainingExecutionContext.addressOffset(leftOperationResult);

        var result = executionContext.allocateBackwardMemory(rows * columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        VectorOperations.subtractVectorFromVector(leftBuffer, leftOffset, expectedProbability, 0,
                resultBuffer, resultOffset, rows * columns);

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public int getForwardMemorySize() {
        return rows * columns + 1;
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
