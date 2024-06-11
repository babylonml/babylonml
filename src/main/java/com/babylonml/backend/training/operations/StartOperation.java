package com.babylonml.backend.training.operations;

public interface StartOperation extends Operation {
    void calculateGradientUpdate();
}
