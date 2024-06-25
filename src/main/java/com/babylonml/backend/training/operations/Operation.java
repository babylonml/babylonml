package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.execution.TensorPointer;
import com.babylonml.backend.training.execution.TrainingExecutionContext;
import it.unimi.dsi.fastutil.ints.IntImmutableList;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.List;

public interface Operation {
    @NonNull
    IntImmutableList getMaxResultShape();

    @NonNull
    TensorPointer forwardPassCalculation();

    @NonNull
    TensorPointer leftBackwardDerivativeChainValue();

    @NonNull
    TensorPointer rightBackwardDerivativeChainValue();

    List<IntImmutableList> getForwardMemoryAllocations();

    @NonNull
    List<IntImmutableList> getBackwardMemoryAllocations();

    @Nullable
    Operation getLeftPreviousOperation();

    @Nullable
    Operation getRightPreviousOperation();

    void setLeftPreviousOperation(@NonNull Operation leftPreviousOperation);

    void setRightPreviousOperation(@NonNull Operation rightPreviousOperation);

    @Nullable
    Operation getNextOperation();

    void setNextOperation(@NonNull Operation nextOperation);

    void clearNextOperation();

    void updateBackwardDerivativeChainValue(@NonNull TensorPointer backwardDerivativeChainValue);

    boolean requiresBackwardDerivativeChainValue();

    void prepareForNextPropagation();

    void startEpochExecution();

    @NonNull
    TrainingExecutionContext getExecutionContext();
}
