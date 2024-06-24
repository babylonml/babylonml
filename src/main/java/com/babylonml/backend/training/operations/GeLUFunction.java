package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.jspecify.annotations.NonNull;

import java.util.Objects;

public final class GeLUFunction extends AbstractOperation {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static final float SCALAR_1 = 0.5f;
    private static final float SCALAR_2 = 1.0f;
    private static final float SCALAR_3 = (float) Math.sqrt(2 / Math.PI);
    private static final float SCALAR_4 = 0.044715f;
    private static final float SCALAR_5 = 3 * SCALAR_4;

    private final int[] maxShape;

    private final boolean requiresDerivativeChainValue;

    @Nullable
    private TensorPointer leftOperandPointer;

    public GeLUFunction(@NonNull Operation leftOperation) {
        super(leftOperation, null);

        this.maxShape = leftOperation.getMaxResultShape();
        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public int @NonNull [] getMaxResultShape() {
        return maxShape;
    }

    @Override
    public @NonNull TensorPointer forwardPassCalculation() {
        Objects.requireNonNull(leftOperation);
        leftOperandPointer = leftOperation.forwardPassCalculation();

        var leftOperandBuffer = executionContext.getMemoryBuffer(leftOperandPointer.pointer());
        var leftOperandOffset = TrainingExecutionContext.addressOffset(leftOperandPointer.pointer());

        var result = executionContext.allocateForwardMemory(this, leftOperandPointer.shape());

        var resultBuffer = executionContext.getMemoryBuffer(result.pointer());
        var resultOffset = TrainingExecutionContext.addressOffset(result.pointer());

        var stride = TensorOperations.stride(leftOperandPointer.shape());

        var loopBound = SPECIES.loopBound(stride);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, leftOperandBuffer, leftOperandOffset + i);
            // GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x^3)))
            var vc = FloatVector.broadcast(SPECIES, SCALAR_1).mul(va).mul(
                    // 1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
                    FloatVector.broadcast(SPECIES, SCALAR_2).add(
                            // tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
                            FloatVector.broadcast(SPECIES, SCALAR_3).mul(
                                    //x + 0.044715 * x^3
                                    va.add(
                                            // 0.044715 * x^3
                                            va.mul(va).mul(va).mul(SCALAR_4)
                                    )

                            ).lanewise(VectorOperators.TANH)
                    )
            );
            vc.intoArray(resultBuffer, resultOffset + i);
        }


        for (int i = loopBound; i < stride; i++) {
            var leftValue = leftOperandBuffer[leftOperandOffset + i];
            // GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x^3)))
            resultBuffer[i + resultOffset] =
                    (float) (leftValue * SCALAR_1 * (SCALAR_2 +
                            Math.tanh(SCALAR_3 *
                                    (leftValue + SCALAR_4 * leftValue * leftValue * leftValue))));
        }

        return result;
    }

    @Override
    public @NonNull TensorPointer leftBackwardDerivativeChainValue() {
        Objects.requireNonNull(derivativeChainPointer);

        var result = executionContext.allocateBackwardMemory(this, derivativeChainPointer.shape());

        var derivativeBuffer = derivativeChainPointer.buffer();
        var derivativeOffset = derivativeChainPointer.offset();

        Objects.requireNonNull(leftOperandPointer);

        var leftBuffer = leftOperandPointer.buffer();
        var leftOffset = leftOperandPointer.offset();

        var resultBuffer = result.buffer();
        var resultOffset = result.offset();

        var size = TensorOperations.stride(leftOperandPointer.shape());

        var loopBound = SPECIES.loopBound(size);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var value = FloatVector.fromArray(SPECIES, leftBuffer, leftOffset + i);

            //h = tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
            var h = FloatVector.broadcast(SPECIES, SCALAR_3).mul(
                    value.add(
                            value.mul(value).mul(value).mul(FloatVector.broadcast(SPECIES, SCALAR_4))
                    )
            ).lanewise(VectorOperators.TANH);
            // d(GeLU(x))/dx = 0.5 * (1 + h + x * (1 - h^2) * (sqrt(2 / PI) + 3 * 0.044715 * x^2))
            var vc = FloatVector.broadcast(SPECIES, SCALAR_1).mul(
                    FloatVector.broadcast(SPECIES, SCALAR_2).add(h).add(
                            //x * (1 - h^2) * (sqrt(2 / PI) + 3 * 0.044715 * x^2))
                            value.mul(
                                    //sqrt(2 / PI) + 3 * 0.044715 * x^2)
                                    FloatVector.broadcast(SPECIES, SCALAR_3).add(
                                            value.mul(value).mul(FloatVector.broadcast(SPECIES, SCALAR_5)
                                            )
                                    ).mul(
                                            //(1 - h^2)
                                            FloatVector.broadcast(SPECIES, SCALAR_2).sub(h.mul(h))
                                    )
                            )
                    )
            );

            var derivative = FloatVector.fromArray(SPECIES, derivativeBuffer, derivativeOffset + i);
            vc = vc.mul(derivative);

            vc.intoArray(resultBuffer, resultOffset + i);
        }

        for (int i = loopBound; i < size; i++) {
            var value = leftBuffer[leftOffset + i];
            //h = tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
            var h = (float)
                    Math.tanh(SCALAR_3 * (value + SCALAR_4 * value * value * value));
            // d(GeLU(x))/dx = 0.5 * (1 + h + x * (1 - h^2) * (sqrt(2 / PI) + 3 * 0.044715 * x^2))
            resultBuffer[i + resultOffset] =
                    (SCALAR_1 *
                            (SCALAR_2 + h + value * (SCALAR_3 +
                                    SCALAR_5 * value * value) * (SCALAR_2 - h * h))) * derivativeBuffer[i
                            + derivativeOffset];
        }

        return result;
    }

    @Override
    public @NonNull TensorPointer rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @NotNull
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

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }
}
