package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import com.babylonml.backend.training.initializer.Initializer;
import com.babylonml.backend.training.optimizer.GradientOptimizer;

import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.Objects;

public final class Variable extends AbstractOperation implements StartOperation {
    @NonNull
    private final GradientOptimizer optimizer;
    private final float learningRate;

    private final float @NonNull [] data;
    private final int[] shape;

    public Variable(@NonNull TrainingExecutionContext executionContext, @NonNull GradientOptimizer optimizer,
                    float @NonNull [] data, int[] shape, float learningRate) {
        this(null, executionContext, optimizer, data, shape, learningRate);
    }

    public Variable(@Nullable String name, @NonNull TrainingExecutionContext executionContext,
                    @NonNull GradientOptimizer optimizer,
                    int[] shape, float learningRate, Initializer initializer) {
        this(name, executionContext, optimizer, initData(shape, initializer),
                shape, learningRate);
    }

    private static float @NonNull [] initData(int[] shape, Initializer initializer) {
        var data = new float[TensorOperations.stride(shape)];
        initializer.initialize(data, 0, shape);

        return data;
    }

    public Variable(@Nullable String name, @NonNull TrainingExecutionContext executionContext,
                    @NonNull GradientOptimizer optimizer,
                    float @NonNull [] data, int[] shape, float learningRate) {
        super(name, executionContext, null, null);

        this.optimizer = optimizer;
        this.data = data;
        this.shape = shape;
        this.learningRate = learningRate;
    }

    @Override
    public int @NonNull [] getMaxResultShape() {
        return shape;
    }

    @Override
    public @NonNull TensorPointer forwardPassCalculation() {
        var result = executionContext.allocateForwardMemory(shape);
        var resultBuffer = executionContext.getMemoryBuffer(result.pointer());
        var resultOffset = TrainingExecutionContext.addressOffset(result.pointer());

        var stride = TensorOperations.stride(shape);
        System.arraycopy(data, 0, resultBuffer, resultOffset, stride);
        return result;
    }

    @Override
    public int @NonNull [][] getForwardMemoryAllocations() {
        return new int[][]{shape};
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
    public int @NonNull [][] getBackwardMemoryAllocations() {
        return optimizer.calculateRequiredMemoryAllocations(shape);
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return true;
    }

    public float @NonNull [] getData() {
        return data;
    }

    @Override
    public void calculateGradientUpdate() {
        Objects.requireNonNull(derivativeChainPointer);

        var derivativeBuffer = executionContext.getMemoryBuffer(derivativeChainPointer.pointer());
        var derivativeOffset = TrainingExecutionContext.addressOffset(derivativeChainPointer.pointer());

        optimizer.optimize(executionContext, data, 0, shape, derivativeBuffer,
                derivativeOffset, learningRate);
    }
}
