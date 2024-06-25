package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.training.execution.TrainingExecutionContext;

import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.List;


@SuppressWarnings("unused")
public final class Constant extends AbstractOperation implements StartOperation {
    private final Tensor constant;

    public Constant(@NonNull TrainingExecutionContext executionContext, Tensor constant) {
        this(null, executionContext, constant);
    }

    public Constant(@Nullable String name, @NonNull TrainingExecutionContext executionContext, Tensor constant) {
        super(name, executionContext, null, null);
        this.constant = constant;
    }

    @Override
    public @NonNull IntImmutableList getMaxResultShape() {
        return constant.getShape();
    }

    @Override
    public @NonNull TensorPointer forwardPassCalculation() {
        var result = executionContext.allocateForwardMemory(this, constant.getShape());

        var stride = TensorOperations.stride(constant.getShape());
        System.arraycopy(constant.getData(), 0, result.buffer(), result.offset(), stride);

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

    @Override
    public @NonNull List<IntImmutableList> getForwardMemoryAllocations() {
        return List.of(constant.getShape());
    }

    @Override
    public @NonNull List<IntImmutableList> getBackwardMemoryAllocations() {
        return List.of();
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
