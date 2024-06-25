package com.babylonml.backend.training.operations;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.execution.TensorPointer;
import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.List;

public class CrossEntropyCostFunction extends AbstractOperation implements CostFunction {

    private final IntImmutableList shape;

    public CrossEntropyCostFunction(@NonNull Operation expectedValues, @NonNull Operation leftOperation) {
        super(leftOperation, expectedValues);

        this.shape = TensorOperations.calculateMaxShape(expectedValues.getMaxResultShape(),
                leftOperation.getMaxResultShape());
    }

    @Override
    public @NonNull IntImmutableList getMaxResultShape() {
        return shape;
    }

    public @Nullable Operation getExpectedValues() {
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

    @Override
    public @NonNull List<IntImmutableList> getForwardMemoryAllocations() {
        throw new UnsupportedOperationException("This is stub class that is used to implement mix of cross entropy" +
                " and softmax. It should not be used in forward pass");
    }

    @Override
    public @NonNull List<IntImmutableList> getBackwardMemoryAllocations() {
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
