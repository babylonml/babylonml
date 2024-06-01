package com.tornadoml.cpu;

import jdk.incubator.vector.*;

import java.util.Arrays;

public final class MatrixOperations {
    private static final float LOG_2 = (float) Math.log(2);
    private static final float LOG_2_E = (float) (1.0 / Math.log(2));

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static void matrixToMatrixMultiplication(float[] firstMatrix, int firstMatrixOffset,
                                                    int firstMatrixRows, int firstMatrixColumns, float[] secondMatrix,
                                                    int secondMatrixOffset, int secondMatrixRows,
                                                    int secondMatrixColumns, float[] result, int resultOffset) {
        assert firstMatrixColumns == secondMatrixRows;
        assert firstMatrix != result;
        assert secondMatrix != result;
        assert firstMatrix != secondMatrix;

        assert firstMatrixRows * firstMatrixColumns + firstMatrixOffset <= firstMatrix.length;
        assert secondMatrixRows * secondMatrixColumns + secondMatrixOffset <= secondMatrix.length;

        assert firstMatrixRows * secondMatrixColumns <= result.length;

        for (int i = 0; i < firstMatrixRows; i++) {
            var resultIndexStart = i * secondMatrixColumns;
            var mulStartIndex = i * firstMatrixColumns;

            for (int j = 0; j < secondMatrixColumns; j += SPECIES.length()) {
                var resultValue = FloatVector.zero(SPECIES);
                //load strip of the second matrix columns
                var secondMatrixMask = SPECIES.indexInRange(j, secondMatrixColumns);

                for (int k = 0, c = 0; k < firstMatrixColumns; k++, c += secondMatrixColumns) {
                    var va = FloatVector.broadcast(SPECIES, firstMatrix[firstMatrixOffset + mulStartIndex + k]);
                    var vb = FloatVector.fromArray(SPECIES, secondMatrix, c + j + secondMatrixOffset, secondMatrixMask);

                    resultValue = va.fma(vb, resultValue);
                }

                var resultValueIndex = resultIndexStart + j;
                resultValue.intoArray(result, resultValueIndex + resultOffset, secondMatrixMask);
            }
        }
    }

    public static void transposeMatrix(float[] matrix, int matrixOffset,
                                       int rows, int columns, float[] result, int resultOffset) {
        assert matrix != result;
        final int blockSize = 16;

        for (int i = 0; i < rows; i += blockSize) {
            for (int j = 0; j < columns; j += blockSize) {
                //use tilling to improve cache locality
                var subRows = Math.min(i + blockSize, rows);
                var subColumns = Math.min(j + blockSize, columns);

                for (int n = i; n < subRows; n++) {
                    for (int k = j; k < subColumns; k++) {
                        result[k * rows + n + resultOffset] = matrix[n * columns + k + matrixOffset];
                    }
                }
            }
        }
    }

    public static void broadcastVectorToMatrix(float[] vector, int vectorOffset, float[] matrix,
                                               int matrixOffset, int rows, int columns) {
        var speciesLength = SPECIES.length();

        for (int i = 0; i < rows; i++) {
            var rowOffset = i * columns + matrixOffset;

            var currentVectorValue = vector[i + vectorOffset];
            var broadCastedValue = FloatVector.broadcast(SPECIES, currentVectorValue);

            var loopBound = SPECIES.loopBound(columns);
            for (int j = 0; j < loopBound; j += speciesLength) {
                broadCastedValue.intoArray(matrix, rowOffset + j);
            }
            for (int j = loopBound; j < columns; j++) {
                matrix[rowOffset + j] = currentVectorValue;
            }
        }
    }

    public static void subMatrix(float[] source, int startColumn, int rows, int columns,
                                 float[] destination, int destinationColumnsCount) {

        for (int i = 0; i < rows; i++) {
            var sourceRowOffset = i * columns + startColumn;
            var destinationRowOffset = i * destinationColumnsCount;

            System.arraycopy(source, sourceRowOffset, destination, destinationRowOffset, destinationColumnsCount);
        }
    }

    public static void reduceMatrixToVector(float[] matrix, int matrixOffset, int rows, int columns, float[] vector, int vectorOffset) {
        for (int i = 0; i < rows; i++) {
            var rowOffset = i * columns + matrixOffset;
            vector[i + vectorOffset] = VectorOperations.sumVectorElements(matrix, rowOffset, columns);
        }
    }

    public static void softMaxByColumns(float[] matrix, int matrixOffset, int rows, int columns, float[] result, int resultOffset) {
        assert matrix != result;
        assert matrix.length - matrixOffset >= result.length - resultOffset;
        assert matrix.length + matrixOffset >= rows * columns;
        assert result.length + resultOffset >= rows * columns;

        //n = round(x * log e)
        //t = x - ln (2) * n (in such case  t lies in the range [-log 2 / 2, log 2 / 2])
        //m = e ^ t
        //e^x = m * 2^n
        //we calculate m and n separately and then combine them to get the result and avoid overflow

        var speciesLength = SPECIES.length();
        var loopBound = SPECIES.loopBound(columns);

        var log2 = FloatVector.broadcast(SPECIES, LOG_2);
        var log2E = FloatVector.broadcast(SPECIES, LOG_2_E);
        var two = FloatVector.broadcast(SPECIES, 2.0f);

        for (int j = 0; j < loopBound; j += speciesLength) {
            var mSum = FloatVector.zero(SPECIES);
            Vector<Float> nSum = FloatVector.broadcast(SPECIES, Float.NEGATIVE_INFINITY);

            for (int i = 0; i < rows; i++) {
                var resultIndex = i * columns + j;
                var columnValues = FloatVector.fromArray(SPECIES, matrix, resultIndex + matrixOffset);

                //n = x * log e
                Vector<Float> nValues = columnValues.mul(log2E);
                //rounding to nearest integer value, n = round(x * log e)
                //extracting the sign of the values and multiplying them by -2, and add 1, to detect sing of 0.5 additivity
                var valuesSigns = nValues.convert(VectorOperators.REINTERPRET_F2I, 0).lanewise(VectorOperators.ASHR, 31).
                        lanewise(VectorOperators.LSHL, 1).
                        add(IntVector.broadcast(IntVector.SPECIES_PREFERRED, 1)).convert(VectorOperators.I2F, 0);
                var roundingAdditivity = FloatVector.broadcast(SPECIES, 0.5f).mul(valuesSigns);
                //rounding to nearest integer value by adding 0.5 and converting to integer and back to float
                nValues = nValues.add(roundingAdditivity).convert(VectorOperators.F2I, 0).convert(VectorOperators.I2F, 0);


                //t = x - ln (2) * n
                var tValues = columnValues.sub(nValues.mul(log2));
                //m = e ^ t
                var mValues = tValues.lanewise(VectorOperators.EXP);

                //nMax = max(n, nSum)
                var nMax = nValues.max(nSum);
                //mSum = m * 2^(n - nMax) + mSum * 2^(nSum - nMax)
                mSum = mValues.mul(two.pow(nValues.sub(nMax))).add(mSum.mul(two.pow(nSum.sub(nMax))));
                nSum = nMax;
            }

            var sumDivider = FloatVector.broadcast(SPECIES, 1.0f).div(mSum);

            for (int i = 0; i < rows; i++) {
                var columnValues = FloatVector.fromArray(SPECIES, matrix, i * columns + j + matrixOffset);

                //n = x * log e
                Vector<Float> nValues = columnValues.mul(log2E);
                //rounding to nearest integer value, n = round(x * log e)
                //extracting the sign of the values and multiplying them by -2, and add 1, to detect sing of 0.5 additivity
                var valuesSigns = nValues.convert(VectorOperators.REINTERPRET_F2I, 0).lanewise(VectorOperators.ASHR, 31).
                        lanewise(VectorOperators.LSHL, 1).
                        add(IntVector.broadcast(IntVector.SPECIES_PREFERRED, 1)).convert(VectorOperators.I2F, 0);
                var roundingAdditivity = FloatVector.broadcast(SPECIES, 0.5f).mul(valuesSigns);
                //rounding to nearest integer value by adding 0.5 and converting to integer and back to float
                nValues = nValues.add(roundingAdditivity).convert(VectorOperators.F2I, 0).convert(VectorOperators.I2F, 0);

                //t = x - ln (2) * n
                var tValues = columnValues.sub(nValues.mul(log2));
                //m = e ^ t
                var mValues = tValues.lanewise(VectorOperators.EXP);

                //result = m * 2^(n - nSum) * sumDivider
                columnValues = mValues.mul(two.pow(nValues.sub(nSum))).mul(sumDivider);

                var resultIndex = i * columns + j;
                columnValues.intoArray(result, resultIndex + resultOffset);
            }
        }

        var reminder = columns - loopBound;
        var mSum = new float[reminder];
        var nSum = new float[reminder];

        Arrays.fill(nSum, Float.NEGATIVE_INFINITY);

        for (int i = 0; i < rows; i++) {
            for (int j = loopBound; j < columns; j++) {
                var sumIndex = j - loopBound;
                var resultIndex = i * columns + j;
                var columnValue = matrix[resultIndex + matrixOffset];

                //n = round(x * log e)
                var nValue = Math.round(columnValue * LOG_2_E);

                //t = x - ln (2) * n
                var tValue = columnValue - LOG_2 * nValue;
                //m = e ^ t
                var mValue = (float) Math.exp(tValue);

                //nMax = max(n, nSum)
                var nMax = Math.max(nValue, nSum[sumIndex]);
                //mSum = m * 2^(n - nMax) + mSum * 2^(nSum - nMax)
                mSum[sumIndex] =
                        (float) (mValue * Math.pow(2, nValue - nMax) +
                                mSum[sumIndex] * Math.pow(2, nSum[sumIndex] - nMax));
                nSum[sumIndex] = nMax;
            }
        }


        for (int i = 0; i < rows; i++) {
            for (int j = loopBound; j < columns; j++) {
                var sumIndex = j - loopBound;
                var resultIndex = i * columns + j;
                var columnValue = matrix[resultIndex + matrixOffset];
                var sumDivider = 1.0f / mSum[sumIndex];

                //n = x * log e
                var nValue = columnValue * LOG_2_E;
                //rounding to nearest integer value, n = round(x * log e)
                var nValueBigPositive = (int) (Math.abs(nValue) + 1);
                nValue = nValue + nValueBigPositive + 0.5f;
                nValue = (int) nValue;

                //t = x - ln (2) * n
                var tValue = columnValue - LOG_2 * nValue;
                //m = e ^ t
                var mValue = (float) Math.exp(tValue);

                //result = m * 2^(n - nSum) * sumDivider
                result[resultIndex + resultOffset] =
                        (float) (mValue * Math.pow(2, nValue - nSum[sumIndex]) * sumDivider);
            }
        }
    }
}
