package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.cpu.VectorOperations;
import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;

import java.util.Objects;


public final class Add extends AbstractOperation {
    private TensorPointer leftOperandPointer;
    private TensorPointer rightOperandPointer;

    private final boolean requiresDerivativeChainValue;

    private final int[] maxShape;

    public Add(Operation leftOperation, Operation rightOperation) {
        this(null, leftOperation, rightOperation);
    }

    public Add(String name, Operation leftOperation, Operation rightOperation) {
        super(name, leftOperation, rightOperation);

        var leftMaxShape = leftOperation.getMaxResultShape();
        var rightMaxShape = rightOperation.getMaxResultShape();

        maxShape = TensorOperations.calculateMaxShape(leftMaxShape, rightMaxShape);

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue()
                || rightOperation.requiresBackwardDerivativeChainValue();
    }


    @Override
    public @NonNull TensorPointer forwardPassCalculation() {
        leftOperandPointer = leftOperation.forwardPassCalculation();
        rightOperandPointer = rightOperation.forwardPassCalculation();

        return broadcastIfNeeded(leftOperandPointer, rightOperandPointer, forwardMemoryAllocator,
                ((firstTensor, secondTensor, result) ->
                        VectorOperations.addVectorToVector(firstTensor.buffer(), firstTensor.offset(), secondTensor.buffer(),
                                secondTensor.offset(),  result.buffer(),
                                result.offset(), TensorOperations.stride(result.shape()))));
    }

    @NotNull
    @Override
    public int @NonNull [][] getForwardMemoryAllocations() {
        return new int[][]{maxShape};
    }

    @Override
    public @NonNull TensorPointer leftBackwardDerivativeChainValue() {
        return calculateDerivative(leftOperandPointer);
    }

    private @NonNull TensorPointer calculateDerivative(@NonNull TensorPointer operandPointer) {
        Objects.requireNonNull(derivativeChainPointer);

        return reduceIfNeeded(operandPointer, derivativeChainPointer, backwardMemoryAllocator,
                ((operand, derivativeChain, result) -> System.arraycopy(derivativeChain.buffer(),
                        derivativeChain.offset(),
                        result.buffer(), result.offset(),
                        TensorOperations.stride(result.shape()))));

    }

    @Override
    public @NonNull TensorPointer rightBackwardDerivativeChainValue() {
        return calculateDerivative(rightOperandPointer);
    }

    @Override
    public int @NonNull [][] getBackwardMemoryAllocations() {
        return new int[][]{maxShape, maxShape};
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }

    @Override
    public int @NonNull [] getMaxResultShape() {
        return maxShape;
    }
}
