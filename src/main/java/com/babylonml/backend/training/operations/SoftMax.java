package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.execution.TensorPointer;
import org.jspecify.annotations.NonNull;

public class SoftMax extends AbstractOperation {
    private final int @NonNull [] maxShape;

    public SoftMax(Operation leftOperation) {
        super(leftOperation, null);

        this.maxShape = leftOperation.getMaxResultShape();
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

    @Override
    public int @NonNull [] getMaxResultShape() {
        return maxShape;
    }

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
}
