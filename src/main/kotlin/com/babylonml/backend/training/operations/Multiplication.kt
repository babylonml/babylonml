package com.babylonml.backend.training.operations

import com.babylonml.backend.cpu.MatrixOperations
import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.cpu.VectorOperations
import com.babylonml.backend.training.execution.TensorPointer
import it.unimi.dsi.fastutil.ints.IntImmutableList
import kotlin.math.max

class Multiplication(name: String?, leftOperation: Operation, rightOperation: Operation) :
    AbstractOperation(name, leftOperation, rightOperation) {

    private var leftOperandResultPointer: TensorPointer? = null
    private var rightOperandResultPointer: TensorPointer? = null

    private val leftTransposeShape: IntImmutableList
    private val rightTransposeShape: IntImmutableList
    private val maxBroadcastShape: IntImmutableList

    constructor(leftOperation: Operation, rightOperation: Operation) : this(
        null,
        leftOperation,
        rightOperation
    )

    init {
        val derivativeShape = calculateResultShape(
            leftOperation.maxResultShape,
            rightOperation.maxResultShape
        )

        val maxLeftResultShape = leftOperation.maxResultShape
        val maxRightResultShape = rightOperation.maxResultShape

        val leftTransposeShapes =
            TensorOperations.broadcastShapes(derivativeShape, maxLeftResultShape, 2)
                ?: throw IllegalArgumentException(
                    "Invalid shapes for multiplication, " +
                            "left operation shape : ${maxLeftResultShape}, " +
                            "right operation: ${maxRightResultShape}."
                )

        if (leftTransposeShapes.first() != derivativeShape) {
            throw IllegalArgumentException(
                "Invalid shapes for multiplication, " +
                        "left operation shape : ${maxLeftResultShape}, " +
                        "right operation: ${maxRightResultShape}."
            )
        }

        leftTransposeShape = leftTransposeShapes.right()

        val rightTransposeShapes = TensorOperations.broadcastShapes(
            derivativeShape,
            maxRightResultShape, 2
        )
        if (rightTransposeShapes == null) {
            throw IllegalArgumentException(
                "Invalid shapes for multiplication, " +
                        "left operation shape : ${maxLeftResultShape}, " +
                        "right operation: ${maxRightResultShape}."
            )
        }

        if (rightTransposeShapes.first() != derivativeShape) {
            throw IllegalArgumentException(
                "Invalid shapes for multiplication, " +
                        "left operation shape : ${maxLeftResultShape}, " +
                        "right operation: ${maxRightResultShape}."
            )
        }

        rightTransposeShape = rightTransposeShapes.right()

        if (leftTransposeShape != maxRightResultShape && maxRightResultShape != rightTransposeShape) {
            val maxBroadcastShapeArray = IntArray(leftTransposeShape.size) {
                max(leftTransposeShape.getInt(it), rightTransposeShape.getInt(it))
            }

            maxBroadcastShape = IntImmutableList.of(*maxBroadcastShapeArray)
        } else if (leftTransposeShape != maxRightResultShape) {
            maxBroadcastShape = maxRightResultShape
        } else if (leftTransposeShape != rightTransposeShape) {
            maxBroadcastShape = leftTransposeShape
        } else {
            maxBroadcastShape = IntImmutableList.of()
        }
    }

    override val maxResultShape: IntImmutableList
            by lazy {
                calculateResultShape(leftOperation.maxResultShape, rightOperation.maxResultShape)
            }

    override fun forwardPassCalculation(): TensorPointer {
        var leftPointer = leftPreviousOperation!!.forwardPassCalculation()
        var rightPointer = rightPreviousOperation!!.forwardPassCalculation()

        leftOperandResultPointer = leftPointer
        rightOperandResultPointer = rightPointer

        val leftShape = leftPointer.shape
        val rightShape = rightPointer.shape

        //If both tensors are 1-dimensional, the dot product (scalar) is returned.
        if (leftShape.size == 1 && rightShape.size == 1) {
            if (leftShape.getInt(0) != rightShape.getInt(0)) {
                throw IllegalStateException(
                    "Incompatible shapes for multiplication: " +
                            "$leftShape and $rightShape"
                )
            }

            val result = executionContext.allocateForwardMemory(this, IntImmutableList.of(1))
            val resultBuffer = result.buffer()
            val resultOffset = result.offset()

            resultBuffer[resultOffset] = VectorOperations.dotProduct(
                leftPointer.buffer(), leftPointer.offset(),
                rightPointer.buffer(), rightPointer.offset(),
                leftShape.getInt(0)
            )

            return result
        }
        //if both arguments are 2-dimensional, the matrix-matrix product is returned.
        if (leftShape.size == 2 && rightShape.size == 2) {
            val result = executionContext.allocateForwardMemory(
                this,
                IntImmutableList.of(leftShape.getInt(0), rightShape.getInt(1))
            )

            val resultBuffer = result.buffer()
            val resultOffset = result.offset()

            MatrixOperations.matrixToMatrixMultiplication(
                leftPointer.buffer(),
                leftPointer.offset(),
                leftShape.getInt(0),
                leftShape.getInt(1),
                rightPointer.buffer(),
                rightPointer.offset(),
                rightShape.getInt(0),
                rightShape.getInt(1),
                resultBuffer,
                resultOffset
            )

            return result
        }

        //If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended
        // to its dimension for the purpose of the matrix multiply.
        if (leftShape.size == 1 && rightShape.size == 2) {
            val result = executionContext.allocateForwardMemory(
                this,
                IntImmutableList.of(1, rightShape.getInt(1))
            )

            val resultBuffer = result.buffer()
            val resultOffset = result.offset()

            MatrixOperations.matrixToMatrixMultiplication(
                leftPointer.buffer(),
                leftPointer.offset(),
                1,
                leftShape.getInt(0),
                rightPointer.buffer(),
                rightPointer.offset(),
                rightShape.getInt(0),
                rightShape.getInt(1),
                resultBuffer,
                resultOffset
            )

            return result
        }

        //If the first argument is 2-dimensional and the second argument is 1-dimensional,
        // the matrix-vector product is returned.
        if (leftShape.size == 2 && rightShape.size == 1) {
            val result = executionContext.allocateForwardMemory(
                this,
                IntImmutableList.of(leftShape.getInt(0), 1)
            )

            val resultBuffer = result.buffer()
            val resultOffset = result.offset()

            MatrixOperations.matrixToMatrixMultiplication(
                leftPointer.buffer(),
                leftPointer.offset(),
                leftShape.getInt(0),
                leftShape.getInt(1),
                rightPointer.buffer(),
                rightPointer.offset(),
                rightShape.getInt(0),
                1,
                resultBuffer,
                resultOffset
            )

            return result
        }

        //If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2),
        //then a batched matrix multiply is returned. If the first argument is 1-dimensional,
        //a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after.
        //If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the
        //batched matrix multiple and removed after.
        val broadcastShapes = TensorOperations.broadcastShapes(leftShape, rightShape, 2)
            ?: throw IllegalStateException(
                "Incompatible shapes for multiplication: " +
                        "$leftShape and $rightShape"
            )

        val broadcastLeft = broadcastShapes.left()
        val broadcastRight = broadcastShapes.right()

        val resultShapeArray = IntArray(broadcastLeft.size) {
            if (it < broadcastLeft.size - 1) {
                broadcastLeft.getInt(it)
            } else {
                broadcastRight.getInt(it)
            }
        }

        resultShapeArray[resultShapeArray.size - 2] = broadcastLeft.getInt(broadcastLeft.size - 2)
        resultShapeArray[resultShapeArray.size - 1] = broadcastRight.getInt(broadcastRight.size - 1)

        if (broadcastLeft != leftShape) {
            val broadcast = executionContext.allocateForwardMemory(this, broadcastLeft)
            TensorOperations.broadcast(
                leftPointer.buffer(),
                leftPointer.offset(),
                leftPointer.shape,
                broadcast.buffer(),
                broadcast.offset(),
                broadcastLeft,
                -1
            )
            leftPointer = broadcast
        } else if (broadcastRight != rightShape) {
            val broadcast = executionContext.allocateForwardMemory(this, broadcastRight)
            TensorOperations.broadcast(
                rightPointer.buffer(),
                rightPointer.offset(),
                rightPointer.shape,
                broadcast.buffer(),
                broadcast.offset(),
                broadcastRight,
                -1
            )
            rightPointer = broadcast
        }

        val resultShape = IntImmutableList.of(*resultShapeArray)
        val result = executionContext.allocateForwardMemory(this, resultShape)
        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        TensorOperations.bmm(
            leftPointer.buffer(), leftPointer.offset(), leftPointer.shape,
            rightPointer.buffer(), rightPointer.offset(), rightPointer.shape,
            resultBuffer, resultOffset, resultShape
        )

        return result
    }

    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(
            maxResultShape, maxBroadcastShape //result and broadcast tensor
        )

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        val rightOperandResultPointer = rightOperandResultPointer!!
        val leftOperandResultPointer = leftOperandResultPointer!!
        val derivativeChainPointer = derivativeChainPointer!!

        val leftShape = leftOperandResultPointer.shape
        val rightShape = rightOperandResultPointer.shape
        val derivativeShape = derivativeChainPointer.shape

        //If both tensors are 1-dimensional, the dot product (scalar) is returned.
        if (leftShape.size == 1 && rightShape.size == 1) {
            if (derivativeShape.size != 1 && derivativeShape.getInt(0) != 1) {
                throw IllegalStateException(
                    "Invalid shapes for left operand or derivative. Left operand: $leftShape  " +
                            "right operand : $rightShape derivative: $derivativeShape."
                )
            }

            return dotProductDerivative(rightOperandResultPointer, derivativeChainPointer)
        }

        //if f both arguments are 2-dimensional, the matrix-matrix product is returned.
        if (leftShape.size == 2 && rightShape.size == 2) {
            if (derivativeShape.size != 2) {
                throw IllegalStateException(
                    "Invalid shapes for left operand or derivative. Left operand: $leftShape  " +
                            "right operand : $rightShape derivative: $derivativeShape."
                )
            }

            return leftOperandMatrixDerivative(rightOperandResultPointer, derivativeChainPointer)
        }

        //If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended
        // to its dimension for the purpose of the matrix multiply.
        if (leftShape.size == 1 && rightShape.size == 2) {
            if (derivativeShape.size != 2) {
                throw IllegalStateException(
                    "Invalid shapes for left operand or derivative. Left operand: $leftShape  " +
                            "right operand : $rightShape derivative: $derivativeShape."
                )
            }

            val result =
                leftOperandMatrixDerivative(rightOperandResultPointer, derivativeChainPointer)
            val resultShape = result.shape
            assert(resultShape.getInt(0) == 1)

            return TensorPointer(
                result.pointer,
                IntImmutableList.of(resultShape.getInt(1)),
                executionContext
            )
        }

        //If the first argument is 2-dimensional and the second argument is 1-dimensional,
        // the matrix-vector product is returned.
        if (leftShape.size == 2 && rightShape.size == 1) {
            if (derivativeShape.size != 2) {
                throw IllegalStateException(
                    "Invalid shapes for left operand or derivative. Left operand: $leftShape  " +
                            "right operand : $rightShape derivative: $derivativeShape."
                )
            }

            return leftOperandMatrixDerivative(rightOperandResultPointer, derivativeChainPointer)
        }

        //If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2),
        //then a batched matrix multiply is returned. If the first argument is 1-dimensional,
        //a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after.
        //If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the
        //batched matrix multiple and removed after.
        val result = leftOperandMatrixDerivative(rightOperandResultPointer, derivativeChainPointer)
        val resultShape = result.shape

        val reducedShapes = TensorOperations.reduceShapes(leftShape, resultShape, 2)
            ?: throw IllegalStateException(
                "Invalid shapes for left operand or derivative. Left operand: $leftShape  " +
                        "right operand : $rightShape derivative: $derivativeShape."
            )


        if (reducedShapes.left() != leftShape) {
            throw IllegalStateException(
                "Invalid shapes for left operand or derivative. Left operand: $leftShape  " +
                        "right operand : $rightShape derivative: $derivativeShape."
            )
        }

        val reducedResultShape = reducedShapes.right()
        if (resultShape == reducedResultShape) {
            return result
        }

        return TensorPointer(
            result.pointer,
            reducedResultShape,
            executionContext
        )
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        val rightOperandResultPointer = rightOperandResultPointer!!
        val leftOperandResultPointer = leftOperandResultPointer!!
        val derivativeChainPointer = derivativeChainPointer!!

        val leftShape = leftOperandResultPointer.shape
        val rightShape = rightOperandResultPointer.shape
        val derivativeShape = derivativeChainPointer.shape

        //If both tensors are 1-dimensional, the dot product (scalar) is returned.
        if (leftShape.size == 1 && rightShape.size == 1) {
            if (derivativeShape.size != 1 && derivativeShape.getInt(0) != 1) {
                throw IllegalStateException(
                    "Invalid shapes for right operand or derivative. Left operand: $leftShape  " +
                            "right operand : $rightShape derivative: $derivativeShape."
                )
            }

            return dotProductDerivative(leftOperandResultPointer, derivativeChainPointer)
        }
        //If the first argument is 2-dimensional and the second argument is 1-dimensional,
        // the matrix-vector product is returned.
        if (leftShape.size == 2 && rightShape.size == 1) {
            if (derivativeShape.size != 2) {
                throw IllegalStateException(
                    "Invalid shapes for right operand or derivative. Left operand: $leftShape  " +
                            "right operand : $rightShape derivative: $derivativeShape."
                )
            }

            val result =
                rightOperandMatrixDerivative(leftOperandResultPointer, derivativeChainPointer)
            assert(result.shape.getInt(1) == 1)

            return TensorPointer(
                result.pointer,
                IntImmutableList.of(result.shape.getInt(0)),
                executionContext
            )
        }

        //If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended
        // to its dimension for the purpose of the matrix multiply.
        if (leftShape.size == 1 && rightShape.size == 2) {
            if (derivativeShape.size != 2) {
                throw IllegalStateException(
                    "Invalid shapes for right operand or derivative. Left operand: $leftShape  " +
                            "right operand : $rightShape derivative: $derivativeShape."
                )
            }

            return rightOperandMatrixDerivative(leftOperandResultPointer, derivativeChainPointer)
        }


        //If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2),
        //then a batched matrix multiply is returned. If the first argument is 1-dimensional,
        //a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after.
        //If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the
        //batched matrix multiple and removed after.

        val result = rightOperandMatrixDerivative(leftOperandResultPointer, derivativeChainPointer)
        val resultShape = result.shape

        val reducedShapes = TensorOperations.reduceShapes(rightShape, resultShape, 2)
            ?: throw IllegalStateException(
                "Invalid shapes for right operand or derivative. Left operand: $leftShape  " +
                        "right operand : $rightShape derivative: $derivativeShape."
            )

        if (reducedShapes.left() != rightShape) {
            throw IllegalStateException(
                "Invalid shapes for right operand or derivative. Left operand: $leftShape  " +
                        "right operand : $rightShape derivative: $derivativeShape."
            )
        }

        val reducedResultShape = reducedShapes.right()
        if (resultShape == reducedResultShape) {
            return result
        }


        return TensorPointer(
            result.pointer,
            reducedResultShape,
            executionContext
        )
    }

    private fun dotProductDerivative(
        operandResultPointer: TensorPointer,
        derivativePointer: TensorPointer
    ): TensorPointer {
        val operandShape = operandResultPointer.shape
        val result = executionContext.allocateBackwardMemory(this, operandShape)

        val scalar = derivativePointer.buffer()[derivativePointer.offset()]
        VectorOperations.multiplyVectorToScalar(
            operandResultPointer.buffer(), operandResultPointer.offset(),
            scalar, result.buffer(), result.offset(), operandShape.getInt(0)
        )

        return result
    }

    private fun leftOperandMatrixDerivative(
        rightOperandResultPointer: TensorPointer,
        derivativeChainPointer: TensorPointer
    ): TensorPointer {
        val rightTransposePointer = transposeOperandResult(
            rightOperandResultPointer,
            derivativeChainPointer
        )

        val derivativeShape = derivativeChainPointer.shape
        val resultShape = TensorOperations.calculateBMMShape(
            derivativeShape, rightTransposePointer.shape
        )

        val result = executionContext.allocateBackwardMemory(this, resultShape)

        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        //leftDerivative = derivative * right^T
        TensorOperations.bmm(
            derivativeChainPointer.buffer(),
            derivativeChainPointer.offset(),
            derivativeShape,
            rightTransposePointer.buffer(),
            rightTransposePointer.offset(),
            rightTransposePointer.shape,
            resultBuffer,
            resultOffset,
            resultShape
        )

        return result

    }

    private fun transposeOperandResult(
        operandResultPointer: TensorPointer,
        derivativeChainPointer: TensorPointer
    ): TensorPointer {
        val rightShape = operandResultPointer.shape
        val derivativeShape = derivativeChainPointer.shape

        val broadcastShapes = TensorOperations.broadcastShapes(rightShape, derivativeShape, 2)
            ?: throw IllegalArgumentException(
                "Incompatible shapes for left operand derivative : " +
                        "${operandResultPointer.shape} and ${derivativeChainPointer.shape}"
            )
        if (broadcastShapes.second() != derivativeShape) {
            throw IllegalArgumentException(
                "Incompatible shapes for left operand derivative : " +
                        "${operandResultPointer.shape} and ${derivativeChainPointer.shape}"
            )
        }

        val broadcastRightPointer = if (rightShape != broadcastShapes.first()) {
            val broadcastShape = broadcastShapes.first()
            val broadcast = executionContext.allocateBackwardMemory(this, broadcastShape)

            TensorOperations.broadcast(
                operandResultPointer.buffer(), operandResultPointer.offset(), rightShape,
                broadcast.buffer(), broadcast.offset(), broadcastShape, -1
            )
            broadcast
        } else {
            operandResultPointer
        }

        val rightTransposeShape = TensorOperations.calculateBMTShape(broadcastRightPointer.shape)
        val transposePointer =
            executionContext.allocateBackwardMemory(this, rightTransposeShape)

        //right^T
        val rightTransposeOffset = transposePointer.offset()
        val rightTransposeBuffer = transposePointer.buffer()

        val broadcastRightBuffer = broadcastRightPointer.buffer()
        val broadcastRightOffset = broadcastRightPointer.offset()
        val broadcastRightShape = broadcastRightPointer.shape

        TensorOperations.bmt(
            broadcastRightBuffer, broadcastRightOffset, broadcastRightShape,
            rightTransposeBuffer, rightTransposeOffset, rightTransposeShape
        )
        return transposePointer
    }

    private fun rightOperandMatrixDerivative(
        leftOperandResultPointer: TensorPointer,
        derivativeChainPointer: TensorPointer
    ): TensorPointer {
        val leftTransposePointer =
            transposeOperandResult(leftOperandResultPointer, derivativeChainPointer)

        val derivativeBuffer = derivativeChainPointer.buffer()
        val derivativeOffset = derivativeChainPointer.offset()

        val leftTransposeShape = leftTransposePointer.shape
        val derivativeShape = derivativeChainPointer.shape

        val resultShape = TensorOperations.calculateBMMShape(leftTransposeShape, derivativeShape)

        val result = executionContext.allocateBackwardMemory(this, resultShape)
        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        //rightDerivative = left^T * derivative
        TensorOperations.bmm(
            leftTransposePointer.buffer(), leftTransposePointer.offset(), leftTransposeShape,
            derivativeBuffer, derivativeOffset, derivativeShape,
            resultBuffer, resultOffset, resultShape
        )

        return result
    }

    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(
            maxBroadcastShape, //broadcast right
            rightTransposeShape,  //right^t
            maxResultShape,  //result derivative

            maxBroadcastShape, //broadcast left
            leftTransposeShape,  //left^t
            maxResultShape //result derivative
        )

    override val requiresBackwardDerivativeChainValue: Boolean by lazy {
        leftPreviousOperation!!.requiresBackwardDerivativeChainValue ||
                rightPreviousOperation!!.requiresBackwardDerivativeChainValue
    }

    companion object {
        private fun calculateResultShape(
            firstShape: IntImmutableList,
            secondShape: IntImmutableList
        ): IntImmutableList {
            //If both tensors are 1-dimensional, the dot product (scalar) is returned.
            if (firstShape.size == 1 && secondShape.size == 1) {
                return IntImmutableList.of(1)
            }
            //if f both arguments are 2-dimensional, the matrix-matrix product is returned.
            if (firstShape.size == 2 && secondShape.size == 2) {
                return IntImmutableList.of(firstShape.getInt(0), secondShape.getInt(1))
            }

            //If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended
            // to its dimension for the purpose of the matrix multiply.
            if (firstShape.size == 1 && secondShape.size == 2) {
                return IntImmutableList.of(1, secondShape.getInt(1))
            }

            //If the first argument is 2-dimensional and the second argument is 1-dimensional,
            // the matrix-vector product is returned.
            if (firstShape.size == 2 && secondShape.size == 1) {
                return IntImmutableList.of(firstShape.getInt(0), 1)
            }

            //If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2),
            //then a batched matrix multiply is returned. If the first argument is 1-dimensional,
            //a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after.
            //If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the
            //batched matrix multiple and removed after.
            val broadcastShapes = TensorOperations.broadcastShapes(firstShape, secondShape, 2)
                ?: throw IllegalArgumentException(
                    "Incompatible shapes for multiplication: " +
                            "$firstShape and $secondShape"
                )

            val modifiedFirstShape = broadcastShapes.left()
            val modifiedSecondShape = broadcastShapes.right()

            val resultShape = IntArray(modifiedFirstShape.size)

            resultShape[modifiedSecondShape.size - 2] =
                modifiedFirstShape.getInt(modifiedFirstShape.size - 2)
            resultShape[modifiedSecondShape.size - 1] =
                modifiedSecondShape.getInt(modifiedSecondShape.size - 1)

            for (i in 0 until modifiedFirstShape.size - 2) {
                resultShape[i] = max(modifiedFirstShape.getInt(i), modifiedSecondShape.getInt(i))
            }

            return IntImmutableList.of(*resultShape)
        }

    }
}
