package com.babylonml.backend.training.operations

import com.babylonml.backend.cpu.TensorOperations
import com.babylonml.backend.training.execution.TensorPointer
import it.unimi.dsi.fastutil.ints.IntImmutableList
import it.unimi.dsi.fastutil.objects.ObjectObjectImmutablePair

class Multiplication(name: String?, leftOperation: Operation, rightOperation: Operation) :
    AbstractOperation(name, leftOperation, rightOperation) {
    private val leftMatrixMaxRows: Int
    private val leftMatrixMaxColumns: Int

    private val rightMatrixMaxRows: Int
    private val rightMatrixMaxColumns: Int

    private var leftOperandResultPointer: TensorPointer? = null
    private var rightOperandResultPointer: TensorPointer? = null

    private val maxOperandShape: IntImmutableList

    constructor(leftOperation: Operation, rightOperation: Operation) : this(null, leftOperation, rightOperation)

    init {
        var leftMaxShape = leftOperation.maxResultShape
        var rightMaxShape = rightOperation.maxResultShape

        val reduceShapes = reduceShapes(leftMaxShape, rightMaxShape)

        leftMaxShape = reduceShapes.left()!!
        rightMaxShape = reduceShapes.right()!!

        this.leftMatrixMaxRows = leftMaxShape.getInt(0)
        this.leftMatrixMaxColumns = leftMaxShape.getInt(1)

        this.rightMatrixMaxRows = rightMaxShape.getInt(0)
        this.rightMatrixMaxColumns = rightMaxShape.getInt(1)

        this.maxOperandShape = TensorOperations.calculateBMMShape(leftMaxShape, rightMaxShape)
    }

    override val maxResultShape: IntImmutableList
        get() = IntImmutableList.of(leftMatrixMaxRows, rightMatrixMaxColumns)

    override fun forwardPassCalculation(): TensorPointer {
        leftOperandResultPointer = leftPreviousOperation!!.forwardPassCalculation()
        rightOperandResultPointer = rightPreviousOperation!!.forwardPassCalculation()

        val leftOperandShape = leftOperandResultPointer!!.shape
        val rightOperandShape = rightOperandResultPointer!!.shape

        val reducedShapes = reduceShapes(leftOperandShape, rightOperandShape)

        val leftShape = reducedShapes.left()!!
        val rightShape = reducedShapes.right()!!

        val resultShape = TensorOperations.calculateBMMShape(leftShape, rightShape)

        val result = executionContext.allocateForwardMemory(this, resultShape)

        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        TensorOperations.bmm(
            leftOperandResultPointer!!.buffer(), leftOperandResultPointer!!.offset(), leftShape,
            rightOperandResultPointer!!.buffer(), rightOperandResultPointer!!.offset(), rightShape,
            resultBuffer, resultOffset, resultShape
        )

        return result
    }

    override val forwardMemoryAllocations: List<IntImmutableList>
        get() = listOf(
            IntImmutableList.of(leftMatrixMaxRows, rightMatrixMaxColumns),
            maxOperandShape
        )

    override fun leftBackwardDerivativeChainValue(): TensorPointer {
        var rightShape = rightOperandResultPointer!!.shape
        var derivativeShape = derivativeChainPointer!!.shape

        val rightBuffer = rightOperandResultPointer!!.buffer()
        val rightOffset = rightOperandResultPointer!!.offset()

        val derivativeBuffer = derivativeChainPointer!!.buffer()
        val derivativeOffset = derivativeChainPointer!!.offset()

        val reducedShapes = reduceShapes(derivativeShape, rightShape)

        derivativeShape = reducedShapes.left()
        rightShape = reducedShapes.right()

        val rightTransposeShape = TensorOperations.calculateBMTShape(rightShape)

        //right^T
        val rightTranspose = executionContext.allocateBackwardMemory(this, rightTransposeShape)
        val rightTransposeOffset = rightTranspose.offset()
        val rightTransposeBuffer = rightTranspose.buffer()

        TensorOperations.bmt(
            rightBuffer, rightOffset, rightShape, rightTransposeBuffer, rightTransposeOffset,
            rightTransposeShape
        )

        val resultShape = TensorOperations.calculateBMMShape(derivativeShape, rightTransposeShape)
        val result = executionContext.allocateBackwardMemory(this, resultShape)

        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        //leftDerivative = derivative * right^T
        TensorOperations.bmm(
            derivativeBuffer, derivativeOffset, derivativeShape,
            rightTransposeBuffer, rightTransposeOffset, rightTransposeShape,
            resultBuffer, resultOffset, resultShape
        )

        return result
    }

    override fun rightBackwardDerivativeChainValue(): TensorPointer {
        var leftShape = leftOperandResultPointer!!.shape

        var derivativeShape = derivativeChainPointer!!.shape
        val reducedShapes = reduceShapes(leftShape, derivativeShape)

        leftShape = reducedShapes.left()
        derivativeShape = reducedShapes.right()

        val leftTransposeShape = TensorOperations.calculateBMTShape(leftShape)

        //left^T
        val leftTranspose = executionContext.allocateBackwardMemory(this, leftTransposeShape)
        val leftTransposeBuffer = leftTranspose.buffer()
        val leftTransposeOffset = leftTranspose.offset()

        val leftBuffer = leftOperandResultPointer!!.buffer()
        val leftOffset = leftOperandResultPointer!!.offset()

        TensorOperations.bmt(
            leftBuffer, leftOffset, leftShape,
            leftTransposeBuffer, leftTransposeOffset, leftTransposeShape
        )

        val derivativeBuffer = derivativeChainPointer!!.buffer()
        val derivativeOffset = derivativeChainPointer!!.offset()

        val resultShape = TensorOperations.calculateBMMShape(leftTransposeShape, derivativeShape)

        val result = executionContext.allocateBackwardMemory(this, resultShape)
        val resultBuffer = result.buffer()
        val resultOffset = result.offset()

        //rightDerivative = left^T * derivative
        TensorOperations.bmm(
            leftTransposeBuffer, leftTransposeOffset, leftTransposeShape,
            derivativeBuffer, derivativeOffset, derivativeShape,
            resultBuffer, resultOffset, resultShape
        )

        return result
    }

    override val backwardMemoryAllocations: List<IntImmutableList>
        get() = listOf( //left
            IntImmutableList.of(leftMatrixMaxRows, leftMatrixMaxColumns),  //right^t
            IntImmutableList.of(rightMatrixMaxColumns, rightMatrixMaxRows),  //right


            IntImmutableList.of(rightMatrixMaxRows, rightMatrixMaxColumns),  //left^t
            IntImmutableList.of(leftMatrixMaxColumns, leftMatrixMaxRows)
        )

    override val requiresBackwardDerivativeChainValue: Boolean by lazy {
        leftPreviousOperation!!.requiresBackwardDerivativeChainValue ||
                rightPreviousOperation!!.requiresBackwardDerivativeChainValue
    }

    companion object {
        private fun reduceShapes(
            firstShape: IntImmutableList?,
            secondShape: IntImmutableList?
        ): ObjectObjectImmutablePair<IntImmutableList?, IntImmutableList?> {
            if (firstShape!!.size == 1 && secondShape!!.size == 1) {
                return ObjectObjectImmutablePair(firstShape, secondShape)
            } else if (firstShape.size < secondShape!!.size) {
                val diff = secondShape.size - firstShape.size
                for (i in 0 until diff) {
                    require(secondShape.getInt(i) == 1) {
                        "Invalid shapes for operation. First shape: " +
                                firstShape + ", second shape: " + secondShape + "."
                    }
                }
                val result = IntArray(firstShape.size)
                secondShape.getElements(diff, result, 0, firstShape.size)
                return ObjectObjectImmutablePair(firstShape, IntImmutableList.of(*result))
            } else {
                val diff = firstShape.size - secondShape.size
                for (i in 0 until diff) {
                    require(firstShape.getInt(i) == 1) {
                        "Invalid shapes for operation. First shape: " +
                                firstShape + ", second shape: " + secondShape + "."
                    }
                }

                val result = IntArray(secondShape.size)
                firstShape.getElements(diff, result, 0, secondShape.size)
                return ObjectObjectImmutablePair(IntImmutableList.of(*result), secondShape)
            }
        }
    }
}
