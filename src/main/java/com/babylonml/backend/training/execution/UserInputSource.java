package com.babylonml.backend.training.execution;

public interface UserInputSource {
    ContextInputSource convertToContextInputSource(int miniBatchSize,
                                                   TrainingExecutionContext executionContext);
}
