package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;
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

    private final int maxRows;
    private final int maxColumns;

    private final boolean requiresDerivativeChainValue;

    private long leftOperandPointer;

    public GeLUFunction(TrainingExecutionContext executionContext, Operation leftOperation) {
        super(executionContext, leftOperation, null);

        this.maxRows = leftOperation.getResultMaxRows();
        this.maxColumns = leftOperation.getResultMaxColumns();

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        leftOperandPointer = leftOperation.forwardPassCalculation();

        var leftOperandBuffer = executionContext.getMemoryBuffer(leftOperandPointer);
        var leftOperandOffset = TrainingExecutionContext.addressOffset(leftOperandPointer);

        var rows = TrainingExecutionContext.rows(leftOperandBuffer, leftOperandOffset);
        var columns = TrainingExecutionContext.columns(leftOperandBuffer, leftOperandOffset);

        assert maxColumns >= columns;
        assert maxRows >= rows;

        var result = executionContext.allocateForwardMemory(rows, columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        var size = rows * columns;


        var loopBound = SPECIES.loopBound(size);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, leftOperandBuffer, leftOperandOffset + i);
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
            var leftValue = leftOperandBuffer[leftOperandOffset + i];
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
        var derivativeBuffer = executionContext.getMemoryBuffer(derivativeChainPointer);
        var derivativeOffset = TrainingExecutionContext.addressOffset(derivativeChainPointer);

        var rows = TrainingExecutionContext.rows(derivativeBuffer, derivativeOffset);
        var columns = TrainingExecutionContext.columns(derivativeBuffer, derivativeOffset);

        assert maxColumns >= columns;
        assert maxRows >= rows;

        var result = executionContext.allocateBackwardMemory(rows, columns);
        var resultBuffer = executionContext.getMemoryBuffer(result);
        var resultOffset = TrainingExecutionContext.addressOffset(result);

        var leftBuffer = executionContext.getMemoryBuffer(leftOperandPointer);
        var leftOffset = TrainingExecutionContext.addressOffset(leftOperandPointer);

        var leftRows = TrainingExecutionContext.rows(leftBuffer, leftOffset);
        var leftColumns = TrainingExecutionContext.columns(leftBuffer, leftOffset);

        assert leftColumns == columns;
        assert leftRows == rows;

        var size = rows * columns;

        var loopBound = SPECIES.loopBound(size);
        for (int i = 0; i < loopBound; i += SPECIES.length()) {
            var value = FloatVector.fromArray(SPECIES, leftBuffer, leftOffset + i);

            //h = tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
            var h = (
                    FloatVector.broadcast(SPECIES, SCALAR_3).mul(
                            value.add(
                                    value.mul(value).mul(value).mul(FloatVector.broadcast(SPECIES, SCALAR_4))
                            )
                    )
            ).lanewise(VectorOperators.TANH);
            // d(GeLU(x))/dx = 0.5 * (1 + h + x * (1 - h^2) * (sqrt(2 / PI) + 3 * 0.044715 * x^2))
            var vc = FloatVector.broadcast(SPECIES, SCALAR_1).mul(
                    FloatVector.broadcast(SPECIES, SCALAR_2).add(h).add(
                            //x * (1 - h^2) * (sqrt(2 / PI) + 3 * 0.044715 * x^2))
                            value.mul(
                                    //sqrt(2 / PI) + 3 * 0.044715 * x^2)
                                    FloatVector.broadcast(SPECIES, SCALAR_3).add(
                                            value.mul(value).mul(FloatVector.broadcast(SPECIES, SCALAR_5)
                                            )
                                    ).mul(
                                            //(1 - h^2)
                                            FloatVector.broadcast(SPECIES, SCALAR_2).sub(h.mul(h))
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
            //h = tanh(sqrt(2 / PI) * (x + 0.044715 * x^3))
            var h = (float)
                    Math.tanh(SCALAR_3 * (value + SCALAR_4 * value * value * value));
            // d(GeLU(x))/dx = 0.5 * (1 + h + x * (1 - h^2) * (sqrt(2 / PI) + 3 * 0.044715 * x^2))
            resultBuffer[i + resultOffset] =
                    (SCALAR_1 *
                            (SCALAR_2 + h + value * (SCALAR_3 +
                                    SCALAR_5 * value * value) * (SCALAR_2 - h * h))) * derivativeBuffer[i
                            + derivativeOffset];
        }

        return result;
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return TrainingExecutionContext.NULL;
    }

    @Override
    public IntIntImmutablePair[] getForwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(maxRows, maxColumns)
        };
    }

    @Override
    public IntIntImmutablePair[] getBackwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(maxRows, maxColumns)
        };
    }

    @Override
    public boolean requiresBackwardDerivativeChainValue() {
        return requiresDerivativeChainValue;
    }

    @Override
    public int getResultMaxRows() {
        return maxRows;
    }

    @Override
    public int getResultMaxColumns() {
        return maxColumns;
    }
}
