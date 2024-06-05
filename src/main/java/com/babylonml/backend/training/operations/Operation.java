package com.babylonml.backend.training.operations;

public interface Operation {
    int getLayerIndex();

    void setLayerIndex(int layerIndex);

    long forwardPassCalculation();

    long leftBackwardDerivativeChainValue();

    long rightBackwardDerivativeChainValue();

    int getForwardMemorySize();

    int getBackwardMemorySize();

    Operation getLeftPreviousOperation();

    Operation getRightPreviousOperation();

    void setLeftPreviousOperation(Operation leftPreviousOperation);

    void setRightPreviousOperation(Operation rightPreviousOperation);

    Operation getNextOperation();

    void setNextOperation(Operation nextOperation);

    void updateBackwardDerivativeChainValue(long backwardDerivativeChainValue);

    boolean requiresBackwardDerivativeChainValue();

    void reset();
}
