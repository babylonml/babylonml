package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

public interface Operation {
    int @NonNull [] getMaxResultShape();

    @NonNull
    TensorPointer forwardPassCalculation();

    @NonNull
    TensorPointer leftBackwardDerivativeChainValue();

    @NonNull
    TensorPointer rightBackwardDerivativeChainValue();

    int @NonNull [][] getForwardMemoryAllocations();

    int @NonNull [][] getBackwardMemoryAllocations();

    @Nullable
    Operation getLeftPreviousOperation();

    @Nullable
    Operation getRightPreviousOperation();

    void setLeftPreviousOperation(@NonNull Operation leftPreviousOperation);

    void setRightPreviousOperation(@NonNull Operation rightPreviousOperation);

    @Nullable
    Operation getNextOperation();

    void setNextOperation(@NonNull Operation nextOperation);

    void updateBackwardDerivativeChainValue(@NonNull TensorPointer backwardDerivativeChainValue);

    boolean requiresBackwardDerivativeChainValue();

    void prepareForNextPropagation();

    void startEpochExecution();

    @NonNull
    TrainingExecutionContext getExecutionContext();
}
