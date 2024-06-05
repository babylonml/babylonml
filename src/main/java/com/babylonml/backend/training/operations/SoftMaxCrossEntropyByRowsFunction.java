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

    private long softMaxResult;

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
        long leftOperationResult = leftOperation.forwardPassCalculation();

        var leftOperationBuffer = executionContext.getMemoryBuffer(leftOperationResult);
        var leftOperationOffset = TrainingExecutionContext.addressOffset(leftOperationResult);

        assert rows * columns == TrainingExecutionContext.addressLength(leftOperationResult);

        softMaxResult = executionContext.allocateForwardMemory(rows * columns);
        var softMaxBuffer = executionContext.getMemoryBuffer(softMaxResult);
        var softMaxOffset = TrainingExecutionContext.addressOffset(softMaxResult);

        MatrixOperations.softMaxByRows(leftOperationBuffer, leftOperationOffset, rows, columns,
                softMaxBuffer, softMaxOffset);

        var loopBound = SPECIES.loopBound(rows * columns);
        var vecSum = FloatVector.zero(SPECIES);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var vec = FloatVector.fromArray(SPECIES, softMaxBuffer,
                    softMaxOffset + i).lanewise(VectorOperators.LOG);
            var expectedVec = FloatVector.fromArray(SPECIES, expectedProbability, i);
            vecSum = vec.fma(expectedVec, vecSum);
        }

        var sum = vecSum.reduceLanes(VectorOperators.ADD);
        for (int i = loopBound; i < rows * columns; i++) {
            sum += (float) Math.log(softMaxBuffer[softMaxOffset + i]) * expectedProbability[i];
        }

        var result = executionContext.allocateForwardMemory(1);

        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);
        resultBuffer[resultOffset] = -sum / columns;

        return result;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        var softMaxBuffer = executionContext.getMemoryBuffer(softMaxResult);
        var softMaxOffset = TrainingExecutionContext.addressOffset(softMaxResult);

        var result = executionContext.allocateBackwardMemory(rows * columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        VectorOperations.subtractVectorFromVector(softMaxBuffer, softMaxOffset, expectedProbability, 0,
                resultBuffer, resultOffset, rows * columns);

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public int getForwardMemorySize() {
        return 2 * rows * columns + 1;
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
