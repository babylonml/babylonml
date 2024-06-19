package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.cpu.MatrixOperations;
import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;

import java.util.Objects;

public final class Multiplication extends AbstractOperation {
    private final int leftMatrixMaxRows;
    private final int leftMatrixMaxColumns;

    private final int rightMatrixMaxRows;
    private final int rightMatrixMaxColumns;

    private TensorPointer leftOperandResultPointer;
    private TensorPointer rightOperandResultPointer;

    private final boolean requiresDerivativeChainValue;

    public Multiplication(Operation leftOperation, Operation rightOperation) {
        this(null, leftOperation, rightOperation);
    }

    public Multiplication(String name, Operation leftOperation, Operation rightOperation) {
        super(name, leftOperation, rightOperation);

        var leftMaxShape = leftOperation.getMaxResultShape();
        var rightMaxShape = rightOperation.getMaxResultShape();

        if (leftMaxShape.length != 2 || rightMaxShape.length != 2) {
            throw new IllegalArgumentException("Multiplication operation can only be performed on matrices");
        }

        this.leftMatrixMaxRows = leftMaxShape[0];
        this.leftMatrixMaxColumns = leftMaxShape[1];

        this.rightMatrixMaxRows = rightMaxShape[0];
        this.rightMatrixMaxColumns = rightMaxShape[1];

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue()
                || rightOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public int @NonNull [] getMaxResultShape() {
        return new int[]{leftMatrixMaxRows, rightMatrixMaxColumns};
    }

    @Override
    public @NotNull TensorPointer forwardPassCalculation() {
        leftOperandResultPointer = leftOperation.forwardPassCalculation();
        var leftOperandBuffer = leftOperandResultPointer.buffer();
        var leftOperandOffset = leftOperandResultPointer.offset();

        rightOperandResultPointer = rightOperation.forwardPassCalculation();
        var rightOperandBuffer = rightOperandResultPointer.buffer();
        var rightOperandOffset = rightOperandResultPointer.offset();

        var leftShape = leftOperandResultPointer.shape();
        var rightShape = rightOperandResultPointer.shape();

        if (leftShape.length != 2 || rightShape.length != 2) {
            throw new IllegalArgumentException("Multiplication operation can only be performed on matrices");
        }

        if (leftShape[1] != rightShape[0]) {
            throw new IllegalArgumentException("Matrix multiplication requires left matrix columns to be equal to right matrix rows");
        }

        var result = executionContext.allocateForwardMemory(leftShape[0], rightShape[1]);
        var resultBuffer = result.buffer();
        var resultOffset = result.offset();

        MatrixOperations.matrixToMatrixMultiplication(leftOperandBuffer, leftOperandOffset, leftShape[0], leftShape[1],
                rightOperandBuffer, rightOperandOffset,
                rightShape[0], rightShape[1],
                resultBuffer, resultOffset);

        return result;
    }

    @Override
    public int @NonNull [][] getForwardMemoryAllocations() {
        return new int[][]{
                new int[]{leftMatrixMaxRows, rightMatrixMaxColumns}
        };
    }

    @Override
    public @NotNull TensorPointer leftBackwardDerivativeChainValue() {
        Objects.requireNonNull(rightOperandResultPointer);
        Objects.requireNonNull(derivativeChainPointer);

        var rightOperandBuffer = rightOperandResultPointer.buffer();
        var rightOperandOffset = rightOperandResultPointer.offset();

        var derivativeBuffer = derivativeChainPointer.buffer();
        var derivativeOffset = derivativeChainPointer.offset();

        var rightShape = rightOperandResultPointer.shape();

        //right^T
        var rightTranspose = executionContext.allocateBackwardMemory(rightShape[1], rightShape[0]);
        var rightTransposeOffset = rightTranspose.offset();
        var rightTransposeBuffer = rightTranspose.buffer();

        MatrixOperations.transposeMatrix(rightOperandBuffer, rightOperandOffset, rightShape[1], rightShape[0],
                rightTransposeBuffer, rightTransposeOffset);


        var derivativeShape = derivativeChainPointer.shape();
        var result = executionContext.allocateBackwardMemory(derivativeShape[0], rightShape[0]);
        var resultBuffer = result.buffer();
        var resultOffset = result.offset();

        //leftDerivative = derivative * right^T
        MatrixOperations.matrixToMatrixMultiplication(derivativeBuffer, derivativeOffset, derivativeShape[0],
                derivativeShape[1],
                rightTransposeBuffer, rightTransposeOffset, rightShape[1], rightShape[0],
                resultBuffer, resultOffset);

        return result;
    }

    @Override
    public @NonNull TensorPointer rightBackwardDerivativeChainValue() {
        Objects.requireNonNull(derivativeChainPointer);
        Objects.requireNonNull(leftOperandResultPointer);

        var leftOperandBuffer = leftOperandResultPointer.buffer();
        var leftOperandOffset = leftOperandResultPointer.offset();

        var leftShape = leftOperandResultPointer.shape();
        //left^T
        var leftTranspose = executionContext.allocateBackwardMemory(leftShape[1], leftShape[0]);
        var leftTransposeBuffer = leftTranspose.buffer();
        var leftTransposeOffset = leftTranspose.offset();

        MatrixOperations.transposeMatrix(leftOperandBuffer, leftOperandOffset, leftShape[0], leftShape[1],
                leftTransposeBuffer, leftTransposeOffset);

        var derivativeBuffer = derivativeChainPointer.buffer();
        var derivativeOffset = derivativeChainPointer.offset();


        var derivativeShape = derivativeChainPointer.shape();

        var result = executionContext.allocateBackwardMemory(leftShape[1], derivativeShape[1]);
        var resultBuffer = result.buffer();
        var resultOffset = result.offset();

        //rightDerivative = left^T * derivative
        MatrixOperations.matrixToMatrixMultiplication(leftTransposeBuffer, leftTransposeOffset, leftShape[1], leftShape[0],
                derivativeBuffer, derivativeOffset, derivativeShape[0], derivativeShape[1],
                resultBuffer, resultOffset);

        return result;
    }

    @Override
    public int @NonNull [][] getBackwardMemoryAllocations() {
        return new int[][]{
                //left
                new int[]{leftMatrixMaxRows, leftMatrixMaxColumns},
                //right^t
                new int[]{rightMatrixMaxColumns, rightMatrixMaxRows},


                //right
                new int[]{rightMatrixMaxRows, rightMatrixMaxColumns},
                //left^t
                new int[]{leftMatrixMaxColumns, leftMatrixMaxRows},
        };
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }
}
