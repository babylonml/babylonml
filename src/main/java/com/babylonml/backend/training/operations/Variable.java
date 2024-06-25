package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import com.babylonml.backend.training.initializer.Initializer;
import com.babylonml.backend.training.optimizer.GradientOptimizer;

import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.List;
import java.util.Objects;

public final class Variable extends AbstractOperation implements StartOperation {
    @NonNull
    private final GradientOptimizer optimizer;
    private final float learningRate;

    private final @NonNull Tensor data;

    public Variable(@NonNull TrainingExecutionContext executionContext, @NonNull GradientOptimizer optimizer,
                    @NonNull Tensor data, float learningRate) {
        this(null, executionContext, optimizer, data, learningRate);
    }

    public Variable(@Nullable String name, @NonNull TrainingExecutionContext executionContext,
                    @NonNull GradientOptimizer optimizer,
                    int[] shape, float learningRate, Initializer initializer) {
        this(name, executionContext, optimizer, initData(shape, initializer),
                learningRate);
    }

    private static @NonNull Tensor initData(int[] shape, Initializer initializer) {
        var data = new float[TensorOperations.stride(shape)];
        initializer.initialize(data, 0, shape);

        return new Tensor(data, shape);
    }

    public Variable(@Nullable String name, @NonNull TrainingExecutionContext executionContext,
                    @NonNull GradientOptimizer optimizer,
                    @NonNull Tensor data, float learningRate) {
        super(name, executionContext, null, null);

        this.optimizer = optimizer;
        this.data = data;
        this.learningRate = learningRate;
    }

    @Override
    public @NonNull IntImmutableList getMaxResultShape() {
        return data.getShape();
    }

    @Override
    public @NonNull TensorPointer forwardPassCalculation() {
        var data = this.data.getData();
        var shape = this.data.getShape();

        var result = executionContext.allocateForwardMemory(this, shape);
        var resultBuffer = executionContext.getMemoryBuffer(result.pointer());
        var resultOffset = TrainingExecutionContext.addressOffset(result.pointer());

        var stride = TensorOperations.stride(shape);
        System.arraycopy(data, 0, resultBuffer, resultOffset, stride);
        return result;
    }

    @Override
    public @NonNull List<IntImmutableList> getForwardMemoryAllocations() {
        return List.of(data.getShape());
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
    public @NonNull List<IntImmutableList> getBackwardMemoryAllocations() {
        return optimizer.calculateRequiredMemoryAllocations(data.getShape());
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return true;
    }

    public float @NonNull [] getData() {
        return data.getData();
    }

    @Override
    public void calculateGradientUpdate() {
        Objects.requireNonNull(derivativeChainPointer);

        var derivativeBuffer = executionContext.getMemoryBuffer(derivativeChainPointer.pointer());
        var derivativeOffset = TrainingExecutionContext.addressOffset(derivativeChainPointer.pointer());

        optimizer.optimize(executionContext, data.getData(), 0, data.getShape(), derivativeBuffer,
                derivativeOffset, learningRate, this);
    }
}
