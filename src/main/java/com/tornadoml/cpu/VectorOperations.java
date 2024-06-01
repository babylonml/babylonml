package com.tornadoml.cpu;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class VectorOperations {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static void multiplyVectorToScalar(float[] vector, int vectorOffset, float scalar, float[] result, int resultOffset,
                                              int length) {
        assert vector.length >= length + vectorOffset;
        assert result.length >= length + resultOffset;

        var speciesLength = SPECIES.length();
        var loopBound = SPECIES.loopBound(length);

        var broadCastedScalar = FloatVector.broadcast(SPECIES, scalar);
        for (int i = 0; i < loopBound; i += speciesLength) {
            var va = FloatVector.fromArray(SPECIES, vector, vectorOffset + i);
            var vc = va.mul(broadCastedScalar);

            vc.intoArray(result, resultOffset + i);
        }

        for (int i = loopBound; i < length; i++) {
            result[resultOffset + i] = vector[vectorOffset + i] * scalar;
        }
    }

    public static void addVectorToVector(float[] first, int firstOffset, float[] second, int secondOffset,
                                         float[] result, int resultOffset, int length) {
        assert first.length >= length;
        assert second.length >= length;

        var speciesLength = SPECIES.length();
        var loopBound = SPECIES.loopBound(length);

        for (int i = 0; i < loopBound; i += speciesLength) {
            var va = FloatVector.fromArray(SPECIES, first, i + firstOffset);
            var vb = FloatVector.fromArray(SPECIES, second, i + secondOffset);
            var vc = va.add(vb);

            vc.intoArray(result, i + resultOffset);
        }

        for (int i = loopBound; i < length; i++) {
            result[i + firstOffset] = first[i + secondOffset] + second[i + resultOffset];
        }
    }

    public static void subtractVectorFromVector(float[] first, int firstOffset, float[] second, int secondOffset,
                                                float[] result, int resultOffset, int length) {
        assert first.length >= length;
        assert second.length >= length;

        var speciesLength = SPECIES.length();
        var loopBound = SPECIES.loopBound(length);

        for (int i = 0; i < loopBound; i += speciesLength) {
            var va = FloatVector.fromArray(SPECIES, first, i + firstOffset);
            var vb = FloatVector.fromArray(SPECIES, second, i + secondOffset);
            var vc = va.sub(vb);

            vc.intoArray(result, i + resultOffset);
        }

        for (int i = loopBound; i < length; i++) {
            result[i + resultOffset] = first[i + firstOffset] - second[i + secondOffset];
        }
    }

    public static void vectorToVectorElementWiseMultiplication(float[] first, int firstOffset,
                                                               float[] second, int secondOffset,
                                                               float[] result, int resultOffset, int length) {
        assert first.length >= length;
        assert second.length >= length;
        assert result.length >= length;

        var speciesLength = SPECIES.length();
        var loopBound = SPECIES.loopBound(length);

        for (int i = 0; i < loopBound; i += speciesLength) {
            var va = FloatVector.fromArray(SPECIES, first, i + firstOffset);
            var vb = FloatVector.fromArray(SPECIES, second, i + secondOffset);
            var vc = va.mul(vb);

            vc.intoArray(result, i + resultOffset);
        }

        for (int i = loopBound; i < length; i++) {
            result[i + resultOffset] = first[i + firstOffset] * second[i + secondOffset];
        }
    }

    public static void vectorElementsSqrt(float[] vector, float[] result, int length) {
        var speciesLength = SPECIES.length();
        var loopBound = SPECIES.loopBound(length);

        for (int i = 0; i < loopBound; i += speciesLength) {
            var va = FloatVector.fromArray(SPECIES, vector, i);
            var vc = va.sqrt();

            vc.intoArray(result, i);
        }

        for (int i = loopBound; i < length; i++) {
            result[i] = (float) Math.sqrt(vector[i]);
        }

    }

    public static void vectorElementsExp(float[] vector, float[] result, int length) {
        assert vector.length >= length;
        assert result.length >= length;

        var speciesLength = SPECIES.length();
        var loopBound = SPECIES.loopBound(length);

        for (int i = 0; i < loopBound; i += speciesLength) {
            var va = FloatVector.fromArray(SPECIES, vector, i);
            var vc = va.lanewise(VectorOperators.EXP);

            vc.intoArray(result, i);
        }

        for (int i = loopBound; i < length; i++) {
            result[i] = (float) Math.exp(vector[i]);
        }
    }

    public static float sumVectorElements(float[] vector, int offset, int length) {
        assert vector.length >= length + offset;

        var speciesLength = SPECIES.length();
        var loopBound = SPECIES.loopBound(length);

        var sum = FloatVector.zero(SPECIES);

        for (int i = 0; i < loopBound; i += speciesLength) {
            var va = FloatVector.fromArray(SPECIES, vector, offset + i);

            sum = sum.add(va);
        }

        var s = sum.reduceLanes(VectorOperators.ADD);
        for (int i = loopBound; i < length; i++) {
            s += vector[offset + i];
        }


        return s;
    }

    public static void addScalarToVector(float scalar, float[] vector, float[] result, int length) {
        assert vector.length >= length;
        assert result.length >= length;

        var speciesLength = SPECIES.length();
        var loopBound = SPECIES.loopBound(length);

        var broadCastedScalar = FloatVector.broadcast(SPECIES, scalar);

        for (int i = 0; i < loopBound; i += speciesLength) {
            var va = FloatVector.fromArray(SPECIES, vector, i);
            var vc = va.add(broadCastedScalar);

            vc.intoArray(result, i);
        }

        for (int i = loopBound; i < length; i++) {
            result[i] = vector[i] + scalar;
        }
    }

    public static void divideScalarOnVectorElements(float scalar, float[] vector, float[] result, int length) {
        assert vector.length >= length;
        assert result.length >= length;

        var speciesLength = SPECIES.length();
        var loopBound = SPECIES.loopBound(length);

        var one = FloatVector.broadcast(SPECIES, scalar);

        for (int i = 0; i < loopBound; i += speciesLength) {
            var va = FloatVector.fromArray(SPECIES, vector, i);
            var vc = one.div(va);

            vc.intoArray(result, i);
        }

        for (int i = loopBound; i < length; i++) {
            result[i] = scalar / vector[i];
        }
    }

    public static void maxBetweenVectorElements(float[] firstVector, float[] secondVector, float[] result, int length) {
        assert firstVector.length >= length;
        assert secondVector.length >= length;
        assert result.length >= length;

        var speciesLength = SPECIES.length();
        var loopBound = SPECIES.loopBound(length);

        for (int i = 0; i < loopBound; i += speciesLength) {
            var va = FloatVector.fromArray(SPECIES, firstVector, i);
            var vb = FloatVector.fromArray(SPECIES, secondVector, i);
            var vc = va.max(vb);

            vc.intoArray(result, i);
        }

        for (int i = loopBound; i < length; i++) {
            result[i] = Math.max(firstVector[i], secondVector[i]);
        }
    }
}
