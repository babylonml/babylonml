package com.babylonml.backend.training;

import com.babylonml.backend.training.operations.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.function.ToIntFunction;

public final class TrainingExecutionContext {
    public static final long NULL = 0;

    private static final int FORWARD_MEMORY_TYPE = 1;

    private static final int MEMORY_TYPE_MASK = ~(Integer.MIN_VALUE >> 2);

    private float[] forwardMemoryBuffer;
    private int forwardMemoryIndex;

    private float[] previousStepBackwardMemoryBuffer;
    private float[] currentStepBackwardMemoryBuffer;

    private int previousBackwardMemoryBufferFlag = 3;
    private int currentBackwardMemoryBufferFlag = 2;

    private int backwardMemoryIndex;

    private final ArrayList<StartOperation> layers = new ArrayList<>();

    /**
     * Last operations for each layer in execution graph.
     */
    private final ArrayList<Operation> lastOperationsInLayers = new ArrayList<>();

    private Operation terminalOperation;

    public void initializeExecution(Operation terminalOperation) {
        this.terminalOperation = terminalOperation;

        //Find last operations for all layers and optimize the execution graph.
        splitExecutionGraphByLayers();

        //For each layer calculate the maximum buffer size needed for the backward and forward  calculation.
        //For forward calculation, the buffer size is the sum of the memory requirements of all operations in the layer.
        //For backward calculation, the buffer size is maximum of the memory requirements of all operations in the layer.
        initializeBuffers();
    }

    private void splitExecutionGraphByLayers() {
        var visitedOperations = new HashSet<Operation>();
        splitExecutionGraphByLayers(terminalOperation, visitedOperations);

        //Optimize the execution graph by collapsing SoftMax and CrossEntropy operations into a single operation.
        optimizeExecutionGraph();

        for (int i = layers.size() - 1; i >= 0; i--) {
            Operation currentOperation = layers.get(i);
            currentOperation.setLayerIndex(i);

            while (currentOperation.getNextOperation() != null) {
                var tmpOperation = currentOperation.getNextOperation();
                var layerIndex = tmpOperation.getLayerIndex();

                if (layerIndex > -1 && layerIndex != i) {
                    break;
                }

                currentOperation = tmpOperation;
                currentOperation.setLayerIndex(i);
            }

            lastOperationsInLayers.add(currentOperation);
        }

        //we added operations in reverse order so correcting that.
        for (int i = 0; i < lastOperationsInLayers.size() / 2; i++) {
            var temp = lastOperationsInLayers.get(i);

            lastOperationsInLayers.set(i, lastOperationsInLayers.get(lastOperationsInLayers.size() - i - 1));
            lastOperationsInLayers.set(lastOperationsInLayers.size() - i - 1, temp);
        }
    }


    /**
     * Split the execution graph into layers starting from the passed operation.
     */
    private void splitExecutionGraphByLayers(Operation operation, HashSet<Operation> visitedOperations) {
        if (visitedOperations.contains(operation)) {
            return;
        }
        visitedOperations.add(operation);

        var previousLeftOperation = operation.getLeftPreviousOperation();
        if (previousLeftOperation != null) {
            if (previousLeftOperation instanceof StartOperation startOperation) {
                layers.add(startOperation);
            }

            splitExecutionGraphByLayers(previousLeftOperation, visitedOperations);
        }


        var previousRightOperation = operation.getRightPreviousOperation();
        if (previousRightOperation != null) {
            if (previousRightOperation instanceof StartOperation startOperation) {
                layers.add(startOperation);
            }

            splitExecutionGraphByLayers(previousRightOperation, visitedOperations);
        }
    }

    public void executePropagation(int maxSteps) {
        for (var i = 0; i < maxSteps; i++) {
            prepareNextPropagationStep();

            executeForwardPropagation();
            executeBackwardPropagation();
        }
    }

    private void prepareNextPropagationStep() {
        forwardMemoryIndex = 0;
        backwardMemoryIndex = 0;
    }

    public long allocateForwardMemory(int length) {
        assert forwardMemoryIndex + length <= forwardMemoryBuffer.length;

        var address = address(FORWARD_MEMORY_TYPE, forwardMemoryIndex, length);
        forwardMemoryIndex += length;

        return address;
    }

    public long allocateBackwardMemory(int length) {
        assert backwardMemoryIndex + length <= currentStepBackwardMemoryBuffer.length;

        var address = address(currentBackwardMemoryBufferFlag, backwardMemoryIndex, length);
        backwardMemoryIndex += length;

        return address;
    }

    public long executeForwardPropagation() {
        return terminalOperation.forwardPassCalculation();
    }

    private void optimizeExecutionGraph() {
        var visitedOperations = new HashSet<Operation>();

        for (var startOperation : layers) {
            collapseSoftMaxCrossEntropy(startOperation, visitedOperations);
        }
    }

    private void collapseSoftMaxCrossEntropy(Operation operation,
                                             HashSet<Operation> visitedOperations) {
        if (!visitedOperations.add(operation)) {
            return;
        }

        var nextTestedOperation = operation.getNextOperation();

        if (operation instanceof SoftMaxByRows softMax) {
            if (nextTestedOperation instanceof CrossEntropyByRowsFunction crossEntropyFunction) {
                var previousOperation = operation.getLeftPreviousOperation();

                var nextOperation = crossEntropyFunction.getNextOperation();
                nextTestedOperation = nextOperation;

                if (softMax.getRows() != crossEntropyFunction.getRows() ||
                        softMax.getColumns() != crossEntropyFunction.getColumns()) {
                    throw new IllegalArgumentException("Softmax and cross entropy should have the same dimensions");
                }

                var softMaxCrossEntropy = new SoftMaxCrossEntropyByRowsFunction(softMax.getRows(),
                        softMax.getColumns(),
                        crossEntropyFunction.getExpectedValues(),
                        this,
                        previousOperation);

                if (nextOperation != null) {
                    var previousLeftNextOperation = nextOperation.getLeftPreviousOperation();
                    var previousRightNextOperation = nextOperation.getRightPreviousOperation();

                    if (previousLeftNextOperation == crossEntropyFunction) {
                        nextOperation.setLeftPreviousOperation(softMaxCrossEntropy);
                    } else if (previousRightNextOperation == crossEntropyFunction) {
                        nextOperation.setRightPreviousOperation(softMaxCrossEntropy);
                    } else {
                        throw new IllegalArgumentException("Operation is not connected to the next operation");
                    }
                } else if (terminalOperation == crossEntropyFunction) {
                    terminalOperation = softMaxCrossEntropy;
                }
            }
        }

        if (nextTestedOperation != null) {
            collapseSoftMaxCrossEntropy(nextTestedOperation, visitedOperations);
        }
    }

    public void executeBackwardPropagation() {
        for (var i = lastOperationsInLayers.size() - 1; i >= 0; i--) {
            backStep(lastOperationsInLayers.get(i));
            swapBackwardMemoryBuffers();
        }
    }

    private void swapBackwardMemoryBuffers() {
        System.arraycopy(currentStepBackwardMemoryBuffer, 0,
                previousStepBackwardMemoryBuffer,
                0, backwardMemoryIndex);

        var tmp = previousBackwardMemoryBufferFlag;
        previousBackwardMemoryBufferFlag = currentBackwardMemoryBufferFlag;
        currentBackwardMemoryBufferFlag = tmp;

        backwardMemoryIndex = 0;
    }

    private void backStep(Operation operation) {
        if (operation instanceof StartOperation startOperation) {
            startOperation.calculateGradientUpdate();
            return;
        }

        var leftOperation = operation.getLeftPreviousOperation();
        var layerIndex = operation.getLayerIndex();

        if (leftOperation != null && leftOperation.requiresBackwardDerivativeChainValue()) {
            var result = operation.leftBackwardDerivativeChainValue();
            leftOperation.updateBackwardDerivativeChainValue(result);

            if (leftOperation.getLayerIndex() == layerIndex) {
                backStep(leftOperation);
            }
        }

        var rightOperation = operation.getRightPreviousOperation();
        if (rightOperation != null && rightOperation.requiresBackwardDerivativeChainValue()) {
            var result = operation.rightBackwardDerivativeChainValue();
            rightOperation.updateBackwardDerivativeChainValue(result);

            if (rightOperation.getLayerIndex() == layerIndex) {
                backStep(rightOperation);
            }
        }
    }


    private void initializeBuffers() {
        var forwardBufferLength = 0;
        var backwardBufferLength = 0;

        var visitedOperationsForward = new HashSet<Operation>();
        var visitedOperationsBackward = new HashSet<Operation>();

        for (var operation : layers) {
            forwardBufferLength += calculateSingleLayerMemoryRequirements(operation,
                    visitedOperationsForward, Operation::getForwardMemorySize);
            backwardBufferLength =
                    Math.max(backwardBufferLength,
                            calculateSingleLayerMemoryRequirements(operation, visitedOperationsBackward,
                                    Operation::getBackwardMemorySize));

            visitedOperationsForward.clear();
            visitedOperationsBackward.clear();
        }

        forwardMemoryBuffer = new float[forwardBufferLength];
        forwardMemoryIndex = 0;

        previousStepBackwardMemoryBuffer = new float[backwardBufferLength];
        currentStepBackwardMemoryBuffer = new float[backwardBufferLength];

        backwardMemoryIndex = 0;
    }

    private int calculateSingleLayerMemoryRequirements(Operation operation,
                                                       HashSet<Operation> visitedOperations,
                                                       ToIntFunction<Operation> memoryCalculator) {
        if (visitedOperations.contains(operation)) {
            return 0;
        }

        visitedOperations.add(operation);
        var layerIndex = operation.getLayerIndex();
        assert layerIndex >= 0;

        var nextOperation = operation.getNextOperation();

        var resultSize = memoryCalculator.applyAsInt(operation);
        if (nextOperation == null) {
            return resultSize;
        }

        var nextResultSize =
                nextOperation.getLayerIndex() != layerIndex
                        ? 0 :
                        calculateSingleLayerMemoryRequirements(nextOperation, visitedOperations, memoryCalculator);
        return resultSize + nextResultSize;
    }


    public static boolean isNull(long address) {
        return address == 0;
    }

    public float[] getMemoryBuffer(long address) {
        var memoryType = memoryType(address);

        return switch (memoryType) {
            case FORWARD -> forwardMemoryBuffer;
            case PREVIOUS_BACKWARD -> previousStepBackwardMemoryBuffer;
            case CURRENT_BACKWARD -> currentStepBackwardMemoryBuffer;
        };
    }

    private MemoryType memoryType(long address) {
        if (isNull(address)) {
            throw new IllegalArgumentException("Provided address is null");
        }

        var memoryType = address >>> 62;
        if (memoryType == FORWARD_MEMORY_TYPE) {
            return MemoryType.FORWARD;
        }

        if (memoryType == currentBackwardMemoryBufferFlag) {
            return MemoryType.CURRENT_BACKWARD;
        }

        return MemoryType.PREVIOUS_BACKWARD;
    }

    public static int addressLength(long address) {
        if (isNull(address)) {
            throw new IllegalArgumentException("Provided address is null");
        }

        return (int) (address >> 32) & MEMORY_TYPE_MASK;
    }

    public static int addressOffset(long address) {
        if (isNull(address)) {
            throw new IllegalArgumentException("Provided address is null");
        }

        return (int) address;
    }

    public static long address(int memoryType, int offset, int length) {
        return ((long) memoryType << 62) | ((long) length << 32) | offset;
    }

    private enum MemoryType {
        FORWARD,
        PREVIOUS_BACKWARD,
        CURRENT_BACKWARD
    }
}


