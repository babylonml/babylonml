package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class LeakyLeRUFunction extends AbstractOperation {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private final int size;
    private final boolean requiresDerivativeChainValue;
    private final float leakyLeRUASlope;

    private long leftValue;

    public LeakyLeRUFunction(int rows, int columns, float leakyLeRUASlope, TrainingExecutionContext executionContext,
                             Operation leftOperation) {
        super(executionContext, leftOperation, null);
        this.leakyLeRUASlope = leakyLeRUASlope;
        this.size = rows * columns;
        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        leftValue = leftOperation.forwardPassCalculation();

        var leftBuffer = executionContext.getMemoryBuffer(leftValue);
        var leftOffset = TrainingExecutionContext.addressOffset(leftValue);

        assert size == TrainingExecutionContext.addressLength(leftValue);

        var result = executionContext.allocateForwardMemory(size);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        var loopBound = SPECIES.loopBound(size);
        var zero = FloatVector.zero(SPECIES);

        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, leftBuffer, leftOffset + i);
            var mask = va.compare(VectorOperators.LT, zero);
            var vc = va.mul(leakyLeRUASlope, mask);
            vc.intoArray(resultBuffer, resultOffset + i);
        }

        for (int i = loopBound; i < size; i++) {
            var leftValue = leftBuffer[leftOffset + i];
            resultBuffer[i + resultOffset] = leftValue > 0 ? leftValue : leakyLeRUASlope * leftValue;
        }

        return result;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        var leftBuffer = executionContext.getMemoryBuffer(leftValue);
        var leftOffset = TrainingExecutionContext.addressOffset(leftValue);

        var result = executionContext.allocateBackwardMemory(size);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        var derivativeChainBuffer = executionContext.getMemoryBuffer(derivativeChainValue);
        var derivativeChainOffset = TrainingExecutionContext.addressOffset(derivativeChainValue);

        var loopBound = SPECIES.loopBound(size);
        var zero = FloatVector.zero(SPECIES);
        var slope = FloatVector.broadcast(SPECIES, leakyLeRUASlope);
        var one = FloatVector.broadcast(SPECIES, 1.0f);

        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, leftBuffer, leftOffset + i);
            var mask = va.compare(VectorOperators.LT, zero);
            var vc = one.mul(slope, mask);

            var diff = FloatVector.fromArray(SPECIES, derivativeChainBuffer, derivativeChainOffset + i);
            vc = vc.mul(diff);

            vc.intoArray(resultBuffer, resultOffset + i);
        }

        for (int i = loopBound; i < size; i++) {
            resultBuffer[i + resultOffset] = (leftBuffer[i + leftOffset] > 0 ? 1.0f : leakyLeRUASlope) *
                    derivativeChainBuffer[i + derivativeChainOffset];
        }

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public int getForwardMemorySize() {
        return size;
    }

    @Override
    public int getBackwardMemorySize() {
        return size;
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }
}
