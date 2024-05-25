package com.tornadoml.cpu;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class MatrixOperations {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static void matrixToMatrixMultiplication(float[] firstMatrix, int firstMatrixOffset,
                                                    int firstMatrixRows, int firstMatrixColumns, float[] secondMatrix,
                                                    int secondMatrixOffset, int secondMatrixRows,
                                                    int secondMatrixColumns, float[] result) {
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
                resultValue.intoArray(result, resultValueIndex, secondMatrixMask);
            }
        }
    }

    public static void transposeMatrix(float[] matrix, int matrixOffset,
                                       int rows, int columns, float[] result) {
        assert matrix != result;
        final int blockSize = 16;

        for (int i = 0; i < rows; i += blockSize) {
            for (int j = 0; j < columns; j += blockSize) {
                //use tilling to improve cache locality
                var subRows = Math.min(i + blockSize, rows);
                var subColumns = Math.min(j + blockSize, columns);

                for (int n = i; n < subRows; n++) {
                    for (int k = j; k < subColumns; k++) {
                        result[k * rows + n] = matrix[n * columns + k + matrixOffset];
                    }
                }
            }
        }
    }

    public static void broadcastVectorToMatrix(float[] vector, float[] matrix, int rows,
                                               int columns) {
        var speciesLength = SPECIES.length();

        for (int i = 0; i < rows; i++) {
            var rowOffset = i * columns;

            var currentVectorValue = vector[i];
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

    public static void reduceMatrixToVector(float[] matrix, int rows, int columns, float[] vector) {
        for (int i = 0; i < rows; i++) {
            var rowOffset = i * columns;
            vector[i] = VectorOperations.sumVectorElements(matrix, rowOffset, columns);
        }
    }

    public static void softMaxByColumns(float[] matrix, int rows, int columns, float[] result) {
        assert matrix != result;
        assert matrix.length >= result.length;
        assert matrix.length >= rows * columns;
        assert result.length >= rows * columns;

        var speciesLength = SPECIES.length();

        VectorOperations.vectorElementsExp(matrix, result, rows * columns);

        var loopBound = SPECIES.loopBound(columns);
        for (int j = 0; j < loopBound; j += speciesLength) {
            var sum = FloatVector.zero(SPECIES);

            for (int i = 0; i < rows; i++) {
                var columnValues = FloatVector.fromArray(SPECIES, result, i * columns + j);
                sum = sum.add(columnValues);
            }
            for (int i = 0; i < rows; i++) {
                var columnValues = FloatVector.fromArray(SPECIES, result, i * columns + j);
                columnValues = columnValues.div(sum);
                columnValues.intoArray(result, i * columns + j);
            }
        }

        var sums = new float[columns - loopBound];
        for (int i = 0; i < rows; i++) {
            var rowOffset = i * columns;
            for (int j = loopBound; j < columns; j++) {
                sums[j - loopBound] += result[rowOffset + j];
            }
        }

        for (int i = 0; i < rows; i++) {
            var rowOffset = i * columns;
            for (int j = loopBound; j < columns; j++) {
                result[rowOffset + j] /= sums[j - loopBound];
            }
        }
    }
}
