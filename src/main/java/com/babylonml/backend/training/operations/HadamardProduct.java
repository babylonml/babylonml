package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.cpu.VectorOperations;


import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.Objects;

public final class HadamardProduct extends AbstractOperation {
    @Nullable
    private TensorPointer leftOperandPointer;

    @Nullable
    private TensorPointer rightOperandPointer;

    private final boolean requiresDerivativeChainValue;

    private final int[] maxShape;

    public HadamardProduct(Operation leftOperation, Operation rightOperation) {
        super(leftOperation, rightOperation);

        this.maxShape = TensorOperations.calculateMaxShape(leftOperation.getMaxResultShape(),
                rightOperation.getMaxResultShape());
        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue() ||
                rightOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public int @NonNull [] getMaxResultShape() {
        return maxShape;
    }

    @Override
    public @NonNull TensorPointer forwardPassCalculation() {
        Objects.requireNonNull(leftOperation);
        Objects.requireNonNull(rightOperation);

        leftOperandPointer = leftOperation.forwardPassCalculation();
        rightOperandPointer = rightOperation.forwardPassCalculation();

        return broadcastIfNeeded(leftOperandPointer, rightOperandPointer, forwardMemoryAllocator,
                (firstTensor, secondTensor, result) ->
                        VectorOperations.vectorToVectorElementWiseMultiplication(firstTensor.buffer(), firstTensor.offset(),
                                secondTensor.buffer(), secondTensor.offset(), result.buffer(), result.offset(),
                                TensorOperations.stride(result.shape())));
    }

    @Override
    public @NonNull TensorPointer leftBackwardDerivativeChainValue() {
        Objects.requireNonNull(derivativeChainPointer);
        Objects.requireNonNull(rightOperandPointer);

        return reduceIfNeeded(derivativeChainPointer, rightOperandPointer, backwardMemoryAllocator,
                (derivativeTensor, rightTensor, resultTensor) ->
                        VectorOperations.vectorToVectorElementWiseMultiplication(

                                derivativeTensor.buffer(), derivativeTensor.offset(),
                                rightTensor.buffer(), rightTensor.offset(),
                                resultTensor.buffer(), resultTensor.offset(),
                                TensorOperations.stride(resultTensor.shape())));
    }

    @Override
    public @NonNull TensorPointer rightBackwardDerivativeChainValue() {
        Objects.requireNonNull(derivativeChainPointer);
        Objects.requireNonNull(leftOperandPointer);

        return reduceIfNeeded(derivativeChainPointer, leftOperandPointer, backwardMemoryAllocator,
                (derivativeTensor, leftTensor, resultTensor) ->
                        VectorOperations.vectorToVectorElementWiseMultiplication(
                                derivativeTensor.buffer(), derivativeTensor.offset(),
                                leftTensor.buffer(), leftTensor.offset(),
                                resultTensor.buffer(), resultTensor.offset(),
                                TensorOperations.stride(resultTensor.shape()))
        );
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
                maxShape,
                maxShape
        };
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }
}
