package com.babylonml.backend.training.operations;

import com.babylonml.backend.training.TrainingExecutionContext;
import com.tornadoml.cpu.MatrixOperations;
import com.tornadoml.cpu.VectorOperations;
import it.unimi.dsi.fastutil.ints.IntIntImmutablePair;


public final class Add extends AbstractOperation {
    private long leftOperandPointer;
    private long rightOperandPointer;

    private final boolean requiresDerivativeChainValue;

    private final int maxRows;
    private final int maxColumns;

    private final boolean broadcast;

    public Add(TrainingExecutionContext executionContext, Operation leftOperation, Operation rightOperation,
               boolean broadcast) {
        this(null, executionContext, leftOperation, rightOperation, broadcast);
    }

    public Add(String name, TrainingExecutionContext executionContext, Operation leftOperation,
               Operation rightOperation, boolean broadcast) {
        super(name, executionContext, leftOperation, rightOperation);

        this.broadcast = broadcast;
        this.maxRows = Math.max(leftOperation.getResultMaxRows(), rightOperation.getResultMaxRows());
        this.maxColumns = Math.max(leftOperation.getResultMaxColumns(), rightOperation.getResultMaxColumns());

        this.requiresDerivativeChainValue = leftOperation.requiresBackwardDerivativeChainValue()
                || rightOperation.requiresBackwardDerivativeChainValue();
    }

    @Override
    public long forwardPassCalculation() {
        leftOperandPointer = leftOperation.forwardPassCalculation();
        var leftResultBuffer = executionContext.getMemoryBuffer(leftOperandPointer);
        var leftResultOffset = TrainingExecutionContext.addressOffset(leftOperandPointer);

        var leftResultRows = TrainingExecutionContext.rows(leftResultBuffer, leftResultOffset);
        var leftResultColumns = TrainingExecutionContext.columns(leftResultBuffer, leftResultOffset);

        rightOperandPointer = rightOperation.forwardPassCalculation();
        var rightResultBuffer = executionContext.getMemoryBuffer(rightOperandPointer);
        var rightResultOffset = TrainingExecutionContext.addressOffset(rightOperandPointer);

        var rightResultRows = TrainingExecutionContext.rows(rightResultBuffer, rightResultOffset);
        var rightResultColumns = TrainingExecutionContext.columns(rightResultBuffer, rightResultOffset);

        if (checkShapes(leftResultRows, rightResultRows, leftResultColumns, rightResultColumns)) {
            var result = executionContext.allocateForwardMemory(leftResultRows, leftResultColumns);
            var resultBuffer = executionContext.getMemoryBuffer(result);
            var resultOffset = TrainingExecutionContext.addressOffset(result);

            VectorOperations.addVectorToVector(leftResultBuffer, leftResultOffset, rightResultBuffer, rightResultOffset,
                    resultBuffer, resultOffset, leftResultRows * leftResultColumns);

            return result;
        } else if (leftResultRows == rightResultRows) {
            var columns = Math.max(leftResultColumns, rightResultColumns);
            var result = executionContext.allocateForwardMemory(leftResultRows, columns);

            var resultBuffer = executionContext.getMemoryBuffer(result);
            var resultOffset = TrainingExecutionContext.addressOffset(result);

            int firsMatrixOffset;
            float[] firstMatrixBuffer;

            int secondMatrixOffset;
            float[] secondMatrixBuffer;


            if (leftResultColumns == 1) {
                firsMatrixOffset = leftResultOffset;
                firstMatrixBuffer = leftResultBuffer;

                secondMatrixOffset = rightResultOffset;
                secondMatrixBuffer = rightResultBuffer;
            } else {
                firsMatrixOffset = rightResultOffset;
                firstMatrixBuffer = rightResultBuffer;

                secondMatrixOffset = leftResultOffset;
                secondMatrixBuffer = leftResultBuffer;
            }

            MatrixOperations.broadcastVectorToMatrixByColumns(firstMatrixBuffer, firsMatrixOffset, resultBuffer,
                    resultOffset, leftResultRows, columns);
            VectorOperations.addVectorToVector(secondMatrixBuffer, secondMatrixOffset, resultBuffer, resultOffset, resultBuffer,
                    resultOffset, leftResultRows * columns);

            return result;
        } else {
            var rows = Math.max(leftResultRows, rightResultRows);

            var result = executionContext.allocateForwardMemory(rows, leftResultColumns);
            var resultBuffer = executionContext.getMemoryBuffer(result);
            var resultOffset = TrainingExecutionContext.addressOffset(result);

            int firsMatrixOffset;
            float[] firstMatrixBuffer;

            int secondMatrixOffset;
            float[] secondMatrixBuffer;

            if (leftResultRows == 1) {
                firsMatrixOffset = leftResultOffset;
                firstMatrixBuffer = leftResultBuffer;

                secondMatrixOffset = rightResultOffset;
                secondMatrixBuffer = rightResultBuffer;
            } else {
                firsMatrixOffset = rightResultOffset;
                firstMatrixBuffer = rightResultBuffer;

                secondMatrixOffset = leftResultOffset;
                secondMatrixBuffer = leftResultBuffer;
            }

            MatrixOperations.broadcastVectorToMatrixByRows(firstMatrixBuffer, firsMatrixOffset, resultBuffer,
                    resultOffset, rows, leftResultColumns);
            VectorOperations.addVectorToVector(secondMatrixBuffer, secondMatrixOffset, resultBuffer, resultOffset, resultBuffer,
                    resultOffset, rows * leftResultColumns);

            return result;
        }
    }

    private boolean checkShapes(int leftResultRows, int rightResultRows,
                                int leftResultColumns, int rightResultColumns) {
        if (leftResultRows == rightResultRows && leftResultColumns == rightResultColumns) {
            return true;
        }

        if (!broadcast) {
            throw new IllegalArgumentException("Invalid shapes for addition operation. Left shape: "
                    + leftResultRows + "x" + leftResultColumns +
                    ", right shape: " + rightResultRows + "x" + rightResultColumns + ".");
        }

        if (leftResultRows == rightResultRows) {
            if (leftResultColumns == 1) {
                return false;
            }

            if (rightResultColumns == 1) {
                return false;
            }
        }

        if (leftResultColumns == rightResultColumns) {
            if (leftResultRows == 1) {
                return false;
            }

            if (rightResultRows == 1) {
                return false;
            }
        }

        throw new IllegalArgumentException("Invalid shapes for addition operation. Left shape: " + leftResultRows + "x" + leftResultColumns +
                ", right shape: " + rightResultRows + "x" + rightResultColumns + ".");
    }

    @Override
    public IntIntImmutablePair[] getForwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(maxRows, maxColumns)
        };
    }

    @Override
    public long leftBackwardDerivativeChainValue() {
        return calculateDerivative(leftOperandPointer);
    }

    private long calculateDerivative(long operandPointer) {
        var derivativeChainBuffer = executionContext.getMemoryBuffer(derivativeChainPointer);
        var derivativeChainValueOffset = TrainingExecutionContext.addressOffset(derivativeChainPointer);

        var derivativeChainRows = TrainingExecutionContext.rows(derivativeChainBuffer, derivativeChainValueOffset);
        var derivativeChainColumns = TrainingExecutionContext.columns(derivativeChainBuffer, derivativeChainValueOffset);

        var operandBuffer = executionContext.getMemoryBuffer(operandPointer);
        var operandOffset = TrainingExecutionContext.addressOffset(operandPointer);

        var operandRows = TrainingExecutionContext.rows(operandBuffer, operandOffset);
        var operandColumns = TrainingExecutionContext.columns(operandBuffer, operandOffset);

        if (operandRows == derivativeChainRows && operandColumns == derivativeChainColumns) {
            var result = executionContext.allocateBackwardMemory(operandRows, operandColumns);
            var resultBuffer = executionContext.getMemoryBuffer(result);
            var resultOffset = TrainingExecutionContext.addressOffset(result);

            System.arraycopy(derivativeChainBuffer, derivativeChainValueOffset, resultBuffer, resultOffset,
                    operandRows * operandColumns);

            return result;
        } else {
            assert operandColumns == 1 || operandRows == 1;

            long result;
            if (operandRows == derivativeChainRows) {
                result = executionContext.allocateBackwardMemory(operandRows, 1);
                var resultBuffer = executionContext.getMemoryBuffer(result);
                var resultOffset = TrainingExecutionContext.addressOffset(result);

                MatrixOperations.reduceMatrixToVectorByColumns(derivativeChainBuffer, derivativeChainValueOffset,
                        derivativeChainRows, derivativeChainColumns,
                        resultBuffer, resultOffset);

            } else {
                result = executionContext.allocateBackwardMemory(1, operandColumns);
                var resultBuffer = executionContext.getMemoryBuffer(result);
                var resultOffset = TrainingExecutionContext.addressOffset(result);

                MatrixOperations.reduceMatrixToVectorByRows(derivativeChainBuffer, derivativeChainValueOffset,
                        derivativeChainRows, derivativeChainColumns, resultBuffer, resultOffset);

            }

            return result;
        }
    }

    @Override
    public long rightBackwardDerivativeChainValue() {
        return calculateDerivative(rightOperandPointer);
    }

    @Override
    public IntIntImmutablePair[] getBackwardMemoryAllocations() {
        return new IntIntImmutablePair[]{
                new IntIntImmutablePair(maxRows, maxColumns),
                new IntIntImmutablePair(maxRows, maxColumns),
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
