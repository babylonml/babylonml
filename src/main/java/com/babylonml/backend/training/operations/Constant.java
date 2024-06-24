package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.training.execution.TrainingExecutionContext;

import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;


@SuppressWarnings("unused")
public final class Constant extends AbstractOperation implements StartOperation {
    private final float[] constant;
    private final int[] shape;


    public Constant(@NonNull TrainingExecutionContext executionContext, float[] constant, int[] shape) {
        this(null, executionContext, constant, shape);
    }

    public Constant(@Nullable String name, @NonNull TrainingExecutionContext executionContext, float[] constant, int[] shape) {
        super(name, executionContext, null, null);
        this.constant = constant;
        this.shape = shape;
    }

    @Override
    public int @NonNull [] getMaxResultShape() {
        return shape;
    }

    @Override
    public @NonNull TensorPointer forwardPassCalculation() {
        var result = executionContext.allocateForwardMemory(this, shape);

        var stride = TensorOperations.stride(shape);
        System.arraycopy(constant, 0, result.buffer(), result.offset(), stride);

        return result;
    }

    @Override
    public @NonNull TensorPointer leftBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public @NonNull TensorPointer rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @NotNull
    @Override
    public int @NonNull [][] getForwardMemoryAllocations() {
        return new int[][]{
                shape
        };
    }

    @Override
    public int @NonNull [][] getBackwardMemoryAllocations() {
        return new int[0][0];
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return false;
    }

    @Override
    public void calculateGradientUpdate() {
        // No gradient update required
    }
}
