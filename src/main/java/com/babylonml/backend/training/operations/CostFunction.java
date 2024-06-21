package com.babylonml.backend.training.operations;

public interface CostFunction extends Operation {
    void trainingMode();
    void fullPassCalculationMode();
}
