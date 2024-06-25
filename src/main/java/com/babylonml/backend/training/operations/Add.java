package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.cpu.VectorOperations;
import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.List;
import java.util.Objects;


public final class Add extends AbstractOperation {
    @Nullable
    private TensorPointer leftOperandPointer;
    @Nullable
    private TensorPointer rightOperandPointer;

    private final boolean requiresDerivativeChainValue;

    private final IntImmutableList maxShape;

    public Add(Operation leftOperation, Operation rightOperation) {
        this(null, leftOperation, rightOperation);
    }

    public Add(@Nullable String name, Operation leftOperation, Operation rightOperation) {
        super(name, leftOperation, rightOperation);

        var leftMaxShape = leftOperation.getMaxResultShape();
        var rightMaxShape = rightOperation.getMaxResultShape();

        maxShape = TensorOperations.calculateMaxShape(leftMaxShape, rightMaxShape);

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue()
                || rightOperation.requiresBackwardDerivativeChainValue();
    }


    @Override
    public @NonNull TensorPointer forwardPassCalculation() {
        Objects.requireNonNull(leftOperation);
        Objects.requireNonNull(rightOperation);

        leftOperandPointer = leftOperation.forwardPassCalculation();
        rightOperandPointer = rightOperation.forwardPassCalculation();

        return broadcastIfNeeded(leftOperandPointer, rightOperandPointer, forwardMemoryAllocator,
                (firstTensor, secondTensor, result) ->
                        VectorOperations.addVectorToVector(firstTensor.buffer(), firstTensor.offset(), secondTensor.buffer(),
                                secondTensor.offset(), result.buffer(),
                                result.offset(), TensorOperations.stride(result.shape())));
    }

    @NotNull
    @Override
    public @NonNull List<IntImmutableList> getForwardMemoryAllocations() {
        return List.of(maxShape);
    }

    @Override
    public @NonNull TensorPointer leftBackwardDerivativeChainValue() {
        Objects.requireNonNull(leftOperandPointer);
        return calculateDerivative(leftOperandPointer);
    }

    private @NonNull TensorPointer calculateDerivative(@NonNull TensorPointer operandPointer) {
        Objects.requireNonNull(derivativeChainPointer);

        return reduceIfNeeded(operandPointer, derivativeChainPointer, backwardMemoryAllocator,
                (operand, derivativeChain, result) -> System.arraycopy(derivativeChain.buffer(),
                        derivativeChain.offset(),
                        result.buffer(), result.offset(),
                        TensorOperations.stride(result.shape())));

    }

    @Override
    public @NonNull TensorPointer rightBackwardDerivativeChainValue() {
        Objects.requireNonNull(rightOperandPointer);
        return calculateDerivative(rightOperandPointer);
    }

    @Override
    public @NonNull List<IntImmutableList> getBackwardMemoryAllocations() {
        return List.of(maxShape, maxShape);
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }

    @Override
    public @NonNull IntImmutableList getMaxResultShape() {
        return maxShape;
    }
}
