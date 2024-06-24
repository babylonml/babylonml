package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.Objects;

public final class MSECostFunction extends AbstractOperation implements CostFunction {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private final int @NonNull [] maxShape;

    private final boolean requiresDerivativeChainValue;

    @Nullable
    private TensorPointer predictionOperandPointer;
    @Nullable
    private TensorPointer expectedValuesPointer;

    private boolean trainingMode;

    public MSECostFunction(Operation predictionOperation,
                           Operation expectedValuesOperation) {
        this(null, predictionOperation, expectedValuesOperation);
    }

    public MSECostFunction(@Nullable String name,
                           Operation predictionOperation,
                           Operation expectedValuesOperation) {
        super(name, predictionOperation, expectedValuesOperation);

        this.maxShape = TensorOperations.calculateMaxShape(predictionOperation.getMaxResultShape(),
                expectedValuesOperation.getMaxResultShape());

        Objects.requireNonNull(leftOperation);
        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public int @NonNull [] getMaxResultShape() {
        return maxShape;
    }

    @Override
    public @NonNull TensorPointer forwardPassCalculation() {
        Objects.requireNonNull(leftOperation);
        Objects.requireNonNull(rightOperation);

        predictionOperandPointer = leftOperation.forwardPassCalculation();
        expectedValuesPointer = rightOperation.forwardPassCalculation();

        if (trainingMode) {
            return TrainingExecutionContext.NULL;
        }

        var predictionOperandBuffer = predictionOperandPointer.buffer();
        var predictionOperandOffset = predictionOperandPointer.offset();

        var expectedValuesBuffer = expectedValuesPointer.buffer();
        var expectedValuesOffset = expectedValuesPointer.offset();

        var result = executionContext.allocateForwardMemory(this, 1, 1);
        var resultBuffer = result.buffer();
        var resultOffset = result.offset();

        var stride = TensorOperations.stride(predictionOperandPointer.shape());
        var loopBound = SPECIES.loopBound(stride);

        var vecSum = FloatVector.zero(SPECIES);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var vec = FloatVector.fromArray(SPECIES, predictionOperandBuffer, predictionOperandOffset + i);
            var expectedVec = FloatVector.fromArray(SPECIES, expectedValuesBuffer, i + expectedValuesOffset);

            var diff = vec.sub(expectedVec);
            vecSum = diff.fma(diff, vecSum);
        }

        var sum = vecSum.reduceLanes(VectorOperators.ADD);
        for (int i = loopBound; i < stride; i++) {
            var value = predictionOperandBuffer[predictionOperandOffset + i] -
                    expectedValuesBuffer[i + expectedValuesOffset];
            sum += value * value;
        }

        resultBuffer[resultOffset] = sum;

        return result;
    }

    @Override
    public @NonNull TensorPointer leftBackwardDerivativeChainValue() {
        Objects.requireNonNull(expectedValuesPointer);
        Objects.requireNonNull(predictionOperandPointer);

        var predictionOperandBuffer = predictionOperandPointer.buffer();
        var predictionOperandOffset = predictionOperandPointer.offset();

        var expectedValuesBuffer = expectedValuesPointer.buffer();
        var expectedValuesOffset = expectedValuesPointer.offset();

        var result = executionContext.allocateBackwardMemory(this, predictionOperandPointer.shape());
        var resultBuffer = result.buffer();
        var resultOffset = result.offset();

        var stride = TensorOperations.stride(predictionOperandPointer.shape());
        var loopBound = SPECIES.loopBound(stride);

        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var vec = FloatVector.fromArray(SPECIES, predictionOperandBuffer, predictionOperandOffset + i);
            var expectedVec = FloatVector.fromArray(SPECIES, expectedValuesBuffer, i + expectedValuesOffset);

            var diff = vec.sub(expectedVec);
            diff.intoArray(resultBuffer, resultOffset + i);
        }

        for (int i = loopBound; i < stride; i++) {
            resultBuffer[i + resultOffset] = predictionOperandBuffer[i + predictionOperandOffset]
                    - expectedValuesBuffer[i + expectedValuesOffset];
        }

        return result;
    }

    @NotNull
    @Override
    public int[][] getForwardMemoryAllocations() {
        return new int[][]{
                new int[]{1, 1}
        };
    }

    @Override
    public int @NonNull [][] getBackwardMemoryAllocations() {
        return new int[][]{
                maxShape
        };
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
    public void trainingMode() {
        trainingMode = true;
    }

    @Override
    public void fullPassCalculationMode() {
        trainingMode = false;
    }
}
