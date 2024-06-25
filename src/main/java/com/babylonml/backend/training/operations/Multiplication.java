package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import it.unimi.dsi.fastutil.ints.IntImmutableList;
import it.unimi.dsi.fastutil.objects.ObjectObjectImmutablePair;
import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.List;
import java.util.Objects;

public final class Multiplication extends AbstractOperation {
    private final int leftMatrixMaxRows;
    private final int leftMatrixMaxColumns;

    private final int rightMatrixMaxRows;
    private final int rightMatrixMaxColumns;

    @Nullable
    private TensorPointer leftOperandResultPointer;
    @Nullable
    private TensorPointer rightOperandResultPointer;

    private final boolean requiresDerivativeChainValue;

    private final IntImmutableList maxOperandShape;

    public Multiplication(Operation leftOperation, Operation rightOperation) {
        this(null, leftOperation, rightOperation);
    }

    public Multiplication(@Nullable String name, Operation leftOperation, Operation rightOperation) {
        super(name, leftOperation, rightOperation);

        var leftMaxShape = leftOperation.getMaxResultShape();
        var rightMaxShape = rightOperation.getMaxResultShape();

        var reduceShapes = reduceShapes(leftMaxShape, rightMaxShape);

        leftMaxShape = reduceShapes.left();
        rightMaxShape = reduceShapes.right();

        this.leftMatrixMaxRows = leftMaxShape.getInt(0);
        this.leftMatrixMaxColumns = leftMaxShape.getInt(1);

        this.rightMatrixMaxRows = rightMaxShape.getInt(0);
        this.rightMatrixMaxColumns = rightMaxShape.getInt(1);

        this.maxOperandShape = TensorOperations.calculateBMMShape(leftMaxShape, rightMaxShape);

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue()
                || rightOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public @NonNull IntImmutableList getMaxResultShape() {
        return IntImmutableList.of(leftMatrixMaxRows, rightMatrixMaxColumns);
    }

    @Override
    public @NotNull TensorPointer forwardPassCalculation() {
        Objects.requireNonNull(leftOperation);
        Objects.requireNonNull(rightOperation);

        leftOperandResultPointer = leftOperation.forwardPassCalculation();
        rightOperandResultPointer = rightOperation.forwardPassCalculation();

        var leftOperandShape = leftOperandResultPointer.shape();
        var rightOperandShape = rightOperandResultPointer.shape();

        var reducedShapes = reduceShapes(leftOperandShape, rightOperandShape);

        var leftShape = reducedShapes.left();
        var rightShape = reducedShapes.right();

        var resultShape = TensorOperations.calculateBMMShape(leftShape, rightShape);

        var result = executionContext.allocateForwardMemory(this, resultShape);

        var resultBuffer = result.buffer();
        var resultOffset = result.offset();

        TensorOperations.bmm(leftOperandResultPointer.buffer(), leftOperandResultPointer.offset(), leftShape,
                rightOperandResultPointer.buffer(), rightOperandResultPointer.offset(), rightShape,
                resultBuffer, resultOffset, resultShape);

        return result;
    }

    @Override
    public @NonNull List<IntImmutableList> getForwardMemoryAllocations() {
        return List.of(
                IntImmutableList.of(leftMatrixMaxRows, rightMatrixMaxColumns),
                maxOperandShape
        );
    }

    @Override
    public @NotNull TensorPointer leftBackwardDerivativeChainValue() {
        Objects.requireNonNull(rightOperandResultPointer);
        Objects.requireNonNull(derivativeChainPointer);

        var rightShape = rightOperandResultPointer.shape();
        var derivativeShape = derivativeChainPointer.shape();

        var rightBuffer = rightOperandResultPointer.buffer();
        var rightOffset = rightOperandResultPointer.offset();

        var derivativeBuffer = derivativeChainPointer.buffer();
        var derivativeOffset = derivativeChainPointer.offset();

        var reducedShapes = reduceShapes(derivativeShape, rightShape);

        derivativeShape = reducedShapes.left();
        rightShape = reducedShapes.right();

        var rightTransposeShape = TensorOperations.calculateBMTShape(rightShape);

        //right^T
        var rightTranspose = executionContext.allocateBackwardMemory(this, rightTransposeShape);
        var rightTransposeOffset = rightTranspose.offset();
        var rightTransposeBuffer = rightTranspose.buffer();

        TensorOperations.bmt(rightBuffer, rightOffset, rightShape, rightTransposeBuffer, rightTransposeOffset,
                rightTransposeShape);

        var resultShape = TensorOperations.calculateBMMShape(derivativeShape, rightTransposeShape);
        var result = executionContext.allocateBackwardMemory(this, resultShape);

        var resultBuffer = result.buffer();
        var resultOffset = result.offset();

        //leftDerivative = derivative * right^T
        TensorOperations.bmm(derivativeBuffer, derivativeOffset, derivativeShape,
                rightTransposeBuffer, rightTransposeOffset, rightTransposeShape,
                resultBuffer, resultOffset, resultShape);

        return result;
    }

    @Override
    public @NonNull TensorPointer rightBackwardDerivativeChainValue() {
        Objects.requireNonNull(derivativeChainPointer);
        Objects.requireNonNull(leftOperandResultPointer);

        var leftShape = leftOperandResultPointer.shape();

        var derivativeShape = derivativeChainPointer.shape();
        var reducedShapes = reduceShapes(leftShape, derivativeShape);

        leftShape = reducedShapes.left();
        derivativeShape = reducedShapes.right();

        var leftTransposeShape = TensorOperations.calculateBMTShape(leftShape);

        //left^T
        var leftTranspose = executionContext.allocateBackwardMemory(this, leftTransposeShape);
        var leftTransposeBuffer = leftTranspose.buffer();
        var leftTransposeOffset = leftTranspose.offset();

        var leftBuffer = leftOperandResultPointer.buffer();
        var leftOffset = leftOperandResultPointer.offset();

        TensorOperations.bmt(leftBuffer, leftOffset, leftShape,
                leftTransposeBuffer, leftTransposeOffset, leftTransposeShape);

        var derivativeBuffer = derivativeChainPointer.buffer();
        var derivativeOffset = derivativeChainPointer.offset();

        var resultShape = TensorOperations.calculateBMMShape(leftTransposeShape, derivativeShape);

        var result = executionContext.allocateBackwardMemory(this, resultShape);
        var resultBuffer = result.buffer();
        var resultOffset = result.offset();

        //rightDerivative = left^T * derivative
        TensorOperations.bmm(leftTransposeBuffer, leftTransposeOffset, leftTransposeShape,
                derivativeBuffer, derivativeOffset, derivativeShape,
                resultBuffer, resultOffset, resultShape);

        return result;
    }

    @Override
    public @NonNull List<IntImmutableList> getBackwardMemoryAllocations() {
        return List.of(
                //left
                IntImmutableList.of(leftMatrixMaxRows, leftMatrixMaxColumns),
                //right^t
                IntImmutableList.of(rightMatrixMaxColumns, rightMatrixMaxRows),


                //right
                IntImmutableList.of(rightMatrixMaxRows, rightMatrixMaxColumns),
                //left^t
                IntImmutableList.of(leftMatrixMaxColumns, leftMatrixMaxRows)
        );
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }

    private static ObjectObjectImmutablePair<IntImmutableList, IntImmutableList> reduceShapes(IntImmutableList firstShape, IntImmutableList secondShape) {
        if (firstShape.size() == 1 && secondShape.size() == 1) {
            return new ObjectObjectImmutablePair<>(firstShape, secondShape);
        } else if (firstShape.size() < secondShape.size()) {
            int diff = secondShape.size() - firstShape.size();
            for (int i = 0; i < diff; i++) {
                if (secondShape.getInt(i) != 1) {
                    throw new IllegalArgumentException("Invalid shapes for operation. First shape: " +
                            firstShape + ", second shape: " + secondShape + ".");
                }
            }
            var result = new int[firstShape.size()];
            secondShape.getElements(diff, result, 0, firstShape.size());
            return new ObjectObjectImmutablePair<>(firstShape, IntImmutableList.of(result));
        } else {
            int diff = firstShape.size() - secondShape.size();
            for (int i = 0; i < diff; i++) {
                if (firstShape.getInt(i) != 1) {
                    throw new IllegalArgumentException("Invalid shapes for operation. First shape: " +
                            firstShape + ", second shape: " + secondShape + ".");
                }
            }

            var result = new int[secondShape.size()];
            firstShape.getElements(diff, result, 0, secondShape.size());
            return new ObjectObjectImmutablePair<>(IntImmutableList.of(result), secondShape);
        }
    }
}
