package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class GeLUFunction extends AbstractOperation {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static final float SCALAR_1 = 0.5f;
    private static final float SCALAR_2 = 1.0f;
    private static final float SCALAR_3 = (float) Math.sqrt(2 / Math.PI);
    private static final float SCALAR_4 = 0.044715f;
    private static final float SCALAR_5 = 3 * SCALAR_4;

    private final int size;

    private final boolean requiresDerivativeChainValue;

    private long leftValue;

    public GeLUFunction(int rows, int columns, TrainingExecutionContext executionContext, Operation leftOperation) {
        super(executionContext, leftOperation, null);
        this.size = rows * columns;
        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        leftValue = leftOperation.forwardPassCalculation();

        var leftBuffer = executionContext.getMemoryBuffer(leftValue);
        var leftOffset = TrainingExecutionContext.addressOffset(leftValue);
        assert size == TrainingExecutionContext.addressLength(leftValue);

        var result = executionContext.allocateForwardMemory(size);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);


        var loopBound = SPECIES.loopBound(size);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, leftBuffer, leftOffset + i);
            // GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x^3)))
            var vc = FloatVector.broadcast(SPECIES, SCALAR_1).mul(va).mul(
                    // 1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
                    FloatVector.broadcast(SPECIES, SCALAR_2).add(
                            // tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
                            FloatVector.broadcast(SPECIES, SCALAR_3).mul(
                                    //x + 0.044715 * x^3
                                    va.add(
                                            // 0.044715 * x^3
                                            va.mul(va).mul(va).mul(SCALAR_4)
                                    )

                            ).lanewise(VectorOperators.TANH)
                    )
            );
            vc.intoArray(resultBuffer, resultOffset + i);
        }


        for (int i = loopBound; i < size; i++) {
            var leftValue = leftBuffer[leftOffset + i];
            // GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x^3)))
            resultBuffer[i + resultOffset] =
                    (float) (leftValue * SCALAR_1 * (SCALAR_2 +
                            Math.tanh(SCALAR_3 *
                                    (leftValue + SCALAR_4 * leftValue * leftValue * leftValue))));
        }

        return result;
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        var derivativeBuffer = executionContext.getMemoryBuffer(derivativeChainValue);
        var derivativeOffset = TrainingExecutionContext.addressOffset(derivativeChainValue);

        var result = executionContext.allocateBackwardMemory(size);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        var leftBuffer = executionContext.getMemoryBuffer(leftValue);
        var leftOffset = TrainingExecutionContext.addressOffset(leftValue);

        var loopBound = SPECIES.loopBound(size);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var value = FloatVector.fromArray(SPECIES, leftBuffer, leftOffset + i);


            // tanh = tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
            var tanh = (value.mul(value.mul(value)).mul(SCALAR_4).add(value)).mul(SCALAR_3).lanewise(VectorOperators.TANH);
            // d(GeLU(x))/dx = 0.5 * (1 + tanh + x * (1 - tanh^2) * (sqrt(2 / PI) + 3 * 0.044715 * x^2))
            var vc = FloatVector.broadcast(SPECIES, SCALAR_1).mul(
                    FloatVector.broadcast(SPECIES, SCALAR_2).add(tanh).add(
                            value.mul(FloatVector.broadcast(SPECIES, SCALAR_3).add(
                                            value.mul(value)).mul(
                                            FloatVector.broadcast(SPECIES, SCALAR_2).sub(tanh.mul(tanh))
                                    )
                            )
                    )
            );

            var derivative = FloatVector.fromArray(SPECIES, derivativeBuffer, derivativeOffset + i);
            vc = vc.mul(derivative);

            vc.intoArray(resultBuffer, resultOffset + i);
        }

        for (int i = loopBound; i < size; i++) {
            var value = leftBuffer[leftOffset + i];
            var tanh = (float)
                    Math.tanh(SCALAR_3 * (value + SCALAR_4 * value * value * value));
            // d(GeLU(x))/dx = 0.5 * (1 + tanh + x * (1 - tanh^2) * (sqrt(2 / PI) + 3 * 0.044715 * x^2))
            resultBuffer[i + resultOffset] =
                    (SCALAR_1 *
                            (SCALAR_2 + tanh + value * (SCALAR_3 +
                                    SCALAR_5 * value * value) * (SCALAR_2 - tanh * tanh))) * derivativeBuffer[i
                            + derivativeOffset];
        }

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public int getForwardMemorySize() {
        return size;
    }

    @Override
    public int getBackwardMemorySize() {
        return size;
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }
}
