package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;

public class CrossEntropyCostFunction extends AbstractOperation implements CostFunction {

    private final int[] shape;

    public CrossEntropyCostFunction(@NonNull Operation expectedValues, @NonNull Operation leftOperation) {
        super(leftOperation, expectedValues);

        this.shape = TensorOperations.calculateMaxShape(expectedValues.getMaxResultShape(),
                leftOperation.getMaxResultShape());
    }

    @Override
    public int @NonNull [] getMaxResultShape() {
        return shape;
    }

    public Operation getExpectedValues() {
        return rightOperation;
    }

    @Override
    public @NonNull TensorPointer forwardPassCalculation() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in forward pass");
    }

    @Override
    public @NonNull TensorPointer leftBackwardDerivativeChainValue() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in backward pass");
    }

    @Override
    public @NonNull TensorPointer rightBackwardDerivativeChainValue() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in backward pass");
    }

    @NotNull
    @Override
    public int @NonNull [][] getForwardMemoryAllocations() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in forward pass");
    }

    @Override
    public int @NonNull [][] getBackwardMemoryAllocations() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in backward pass");
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return false;
    }

    @Override
    public void trainingMode() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in backward pass");
    }

    @Override
    public void fullPassCalculationMode() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in forward pass");
    }
}
