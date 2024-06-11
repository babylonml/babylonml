package com.babylonml.backend.training.operations;

import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;

public interface Operation {
    int getResultMaxRows();

    int getResultMaxColumns();

    int getLayerIndex();

    void setLayerIndex(int layerIndex);

    long forwardPassCalculation();

    long leftBackwardDerivativeChainValue();

    long rightBackwardDerivativeChainValue();

    IntIntImmutablePair[] getForwardMemoryAllocations();

    IntIntImmutablePair[] getBackwardMemoryAllocations();

    Operation getLeftPreviousOperation();

    Operation getRightPreviousOperation();

    void setLeftPreviousOperation(Operation leftPreviousOperation);

    void setRightPreviousOperation(Operation rightPreviousOperation);

    Operation getNextOperation();

    void setNextOperation(Operation nextOperation);

    void updateBackwardDerivativeChainValue(long backwardDerivativeChainValue);

    boolean requiresBackwardDerivativeChainValue();

    void prepareForNextPropagation();
}
