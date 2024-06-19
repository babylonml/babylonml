package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.jspecify.annotations.NonNull;

import java.util.Objects;

public final class LeakyLeRUFunction extends AbstractOperation {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private final int @NonNull [] maxShape;

    private final boolean requiresDerivativeChainValue;
    private final float leakyLeRUASlope;

    private TensorPointer leftOperandResult;

    public LeakyLeRUFunction(float leakyLeRUASlope, Operation leftOperation) {
        this(null, leakyLeRUASlope, leftOperation);
    }

    public LeakyLeRUFunction(String name, float leakyLeRUASlope,
                             Operation leftOperation) {
        super(name, leftOperation, null);
        this.leakyLeRUASlope = leakyLeRUASlope;

        this.maxShape = TensorOperations.calculateMaxShape(leftOperation.getMaxResultShape(),
                leftOperation.getMaxResultShape());

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public int @NonNull [] getMaxResultShape() {
        return maxShape;
    }

    @Override
    public @NonNull TensorPointer forwardPassCalculation() {
        leftOperandResult = leftOperation.forwardPassCalculation();
        var result = executionContext.allocateForwardMemory(leftOperandResult.shape());

        var leftResultBuffer = leftOperandResult.buffer();
        var leftResultOffset = leftOperandResult.offset();

        var resultBuffer = result.buffer();
        var resultOffset = result.offset();

        var size = TensorOperations.stride(leftOperandResult.shape());
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
    public @NonNull TensorPointer leftBackwardDerivativeChainValue() {
        Objects.requireNonNull(derivativeChainPointer);

        var leftOperandBuffer = leftOperandResult.buffer();
        var leftOperandOffset = leftOperandResult.offset();

        var derivativeChainBuffer = derivativeChainPointer.buffer();
        var derivativeChainOffset = derivativeChainPointer.offset();

        var result = executionContext.allocateBackwardMemory(leftOperandResult.shape());

        var resultBuffer = result.buffer();
        var resultOffset = result.offset();

        var size = TensorOperations.stride(leftOperandResult.shape());

        var loopBound = SPECIES.loopBound(size);
        var zero = FloatVector.zero(SPECIES);
        var slope = FloatVector.broadcast(SPECIES, leakyLeRUASlope);
        var one = FloatVector.broadcast(SPECIES, 1.0f);

        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, leftOperandBuffer, leftOperandOffset + i);
            var mask = va.compare(VectorOperators.LT, zero);
            var vc = one.mul(slope, mask);

            var diff = FloatVector.fromArray(SPECIES, derivativeChainBuffer, derivativeChainOffset + i);
            vc = vc.mul(diff);

            vc.intoArray(resultBuffer, resultOffset + i);
        }

        for (int i = loopBound; i < size; i++) {
            resultBuffer[i + resultOffset] = (leftOperandBuffer[i + leftOperandOffset] > 0 ? 1.0f : leakyLeRUASlope) *
                    derivativeChainBuffer[i + derivativeChainOffset];
        }

        return result;
    }

    @Override
    public @NonNull TensorPointer rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }


    @Override
    public int @NonNull [][] getForwardMemoryAllocations() {
        return new int[][]{
                maxShape
        };
    }

    @Override
    public int @NonNull [][] getBackwardMemoryAllocations() {
        return new int[][]{
                maxShape
        };
    }
}
