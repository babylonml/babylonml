package com.babylonml.backend.training.execution;

import com.babylonml.backend.cpu.TensorOperations;
import com.babylonml.backend.training.operations.*;
import org.jspecify.annotations.NonNull;
import org.jspecify.annotations.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;

public final class TrainingExecutionContext {
    public static final TensorPointer NULL = new TensorPointer(0, new int[0], null);

    private static final int FORWARD_MEMORY_TYPE = 1;

    private float[] forwardMemoryBuffer;
    private int forwardMemoryIndex;

    private float[] previousStepBackwardMemoryBuffer;
    private float[] currentStepBackwardMemoryBuffer;

    private int previousBackwardMemoryBufferFlag = 3;
    private int currentBackwardMemoryBufferFlag = 2;

    private int backwardMemoryIndex;

    private final ArrayList<ArrayList<Operation>> layers = new ArrayList<>();

    private CostFunction terminalOperation;

    private final int epochs;
    private final int miniBatchSize;

    private ContextInputSource inputSource;

    private final boolean trackMemoryAllocation;

    private final IdentityHashMap<Operation, long[]> consumedBackwardMemory = new IdentityHashMap<>();
    private final IdentityHashMap<Operation, long[]> consumedForwardMemory = new IdentityHashMap<>();

    /**
     * Create a new training execution context.
     *
     * @param epochs        Number of epochs to train.
     * @param miniBatchSize Size of the mini-batch. If set to -1,
     *                      the mini-batch size will be equal to the number of samples.
     */
    public TrainingExecutionContext(int epochs, int miniBatchSize) {
        this.epochs = epochs;
        this.miniBatchSize = miniBatchSize;
        this.trackMemoryAllocation = false;
    }

    /**
     * Create a new training execution context.
     *
     * @param epochs                Number of epochs to train.
     * @param miniBatchSize         Size of the mini-batch.
     *                              If set to -1, the mini-batch size will be equal to the number of samples.
     * @param trackMemoryAllocation If set to true, the memory allocation of each operation will be tracked.
     */
    public TrainingExecutionContext(int epochs, int miniBatchSize, boolean trackMemoryAllocation) {
        this.epochs = epochs;
        this.miniBatchSize = miniBatchSize;
        this.trackMemoryAllocation = trackMemoryAllocation;
    }

    /**
     * Create a new training execution context.
     * The mini-batch size will be equal to the number of samples.
     *
     * @param epochs Number of epochs to train.
     */
    public TrainingExecutionContext(int epochs) {
        this.epochs = epochs;
        this.miniBatchSize = -1;
        this.trackMemoryAllocation = false;
    }

    /**
     * Create a new training execution context.
     * The mini-batch size will be equal to the number of samples.
     *
     * @param epochs                Number of epochs to train.
     * @param trackMemoryAllocation If set to true, the memory allocation of each operation will be tracked.
     */
    public TrainingExecutionContext(int epochs, boolean trackMemoryAllocation) {
        this.epochs = epochs;
        this.miniBatchSize = -1;
        this.trackMemoryAllocation = trackMemoryAllocation;
    }

    public ContextInputSource registerMainInputSource(Tensor data) {
        if (inputSource != null) {
            throw new IllegalStateException("Input source is already registered");
        }

        this.inputSource =
                new MiniBatchTensorInputDataSource(data).convertToContextInputSource(calculateMiniBatchSize(data),
                        this);

        return inputSource;
    }

    public ContextInputSource registerAdditionalInputSource(Tensor data) {
        if (inputSource == null) {
            throw new IllegalStateException("Main input source is not registered");
        }

        if (data.getShape()[0] != inputSource.getSamplesCount()) {
            throw new IllegalArgumentException("Samples count do not match the main input source");
        }

        return new MiniBatchTensorInputDataSource(data).convertToContextInputSource(calculateMiniBatchSize(data),
                this);
    }

    private int calculateMiniBatchSize(Tensor data) {
        int batchSize;
        if (miniBatchSize == -1) {
            batchSize = data.getShape()[0];
        } else {
            batchSize = miniBatchSize;
        }

        return batchSize;
    }

    public void initializeExecution(CostFunction terminalOperation) {
        this.terminalOperation = terminalOperation;

        //Find last operations for all layers and optimize the execution graph.
        splitExecutionGraphByLayers();

        //For each layer calculate the maximum buffer size needed for the backward and forward  calculation.
        //For forward calculation, the buffer size is the sum of the memory requirements of all operations in the layer.
        //For backward calculation, the buffer size is maximum of the memory requirements of all operations in the layer.
        initializeBuffers();
    }

    private void splitExecutionGraphByLayers() {
        //Optimize the execution graph by collapsing SoftMax and CrossEntropy operations into a single operation.
        optimizeExecutionGraph();

        var operations = new ArrayList<Operation>();
        operations.add(terminalOperation);

        splitExecutionGraphByLayers(operations);

        Collections.reverse(layers);
    }


    /**
     * Split the execution graph into layers starting from the passed operations.
     */
    private void splitExecutionGraphByLayers(ArrayList<Operation> operations) {
        layers.add(operations);

        var nextLayerOperations = new ArrayList<Operation>();
        var visitedOperations = new HashSet<Operation>();
        for (var operation : operations) {
            var previousLeftOperation = operation.getLeftPreviousOperation();
            var previousRightOperation = operation.getRightPreviousOperation();

            if (previousLeftOperation != null) {
                if (visitedOperations.add(previousLeftOperation)) {
                    nextLayerOperations.add(previousLeftOperation);
                }
            }

            if (previousRightOperation != null) {
                if (visitedOperations.add(previousRightOperation)) {
                    nextLayerOperations.add(previousRightOperation);
                }
            }
        }

        if (nextLayerOperations.isEmpty()) {
            return;
        }

        splitExecutionGraphByLayers(nextLayerOperations);
    }

    public void executePropagation() {
        executePropagation(null);
    }

    public void executePropagation(@Nullable EpochCompletionCallback callback) {
        if (callback != null) {
            terminalOperation.fullPassCalculationMode();

            var cost = calculateFullCost();
            callback.onEpochCompleted(0, cost);
        }

        if (inputSource == null) {
            throw new IllegalStateException("Input source is not registered");
        }

        var dataSize = inputSource.getSamplesCount();
        var miniBatchSize = inputSource.getMiniBatchSize();
        int miniBatchCount = (dataSize + miniBatchSize - 1) / miniBatchSize;


        terminalOperation.trainingMode();
        for (int epoch = 0; epoch < epochs; epoch++) {
            terminalOperation.startEpochExecution();

            for (int j = 0; j < miniBatchCount; j++) {
                prepareNextPropagationStep();

                assert inputSource == null || inputSource.gitLocalMiniBatchIndex() == j;

                executeForwardPropagation();
                executeBackwardPropagation();
            }

            if (callback != null) {
                terminalOperation.fullPassCalculationMode();

                var cost = calculateFullCost();
                callback.onEpochCompleted(epoch + 1, cost);
            }
        }
    }

    private float calculateFullCost() {
        var dataSize = inputSource.getSamplesCount();
        var miniBatchCount = (dataSize + miniBatchSize - 1) / miniBatchSize;

        terminalOperation.startEpochExecution();
        var sum = 0.0f;

        for (int j = 0; j < miniBatchCount; j++) {
            prepareNextPropagationStep();

            var result = executeForwardPropagation();
            var resultBuffer = result.buffer();

            assert TensorOperations.stride(result.shape()) == 1;

            sum += resultBuffer[result.offset()];
        }

        return sum / dataSize;
    }


    private void prepareNextPropagationStep() {
        forwardMemoryIndex = 0;
        backwardMemoryIndex = 0;

        consumedForwardMemory.clear();
        consumedBackwardMemory.clear();

        terminalOperation.prepareForNextPropagation();
    }


    public TensorPointer allocateForwardMemory(Operation operation, int... dimensions) {
        var length = 1;
        for (var dimension : dimensions) {
            length *= dimension;
        }

        if (trackMemoryAllocation) {
            var allocationsSize = allocationsSize(operation.getForwardMemoryAllocations());
            var allocated = consumedForwardMemory.computeIfAbsent(operation, (k) -> new long[1]);

            if (length + allocated[0] > allocationsSize) {
                throw new IllegalStateException("Memory allocation exceeded the required memory size for operation "
                        + operation);
            }

            allocated[0] += length;
        }

        assert forwardMemoryIndex + length <= forwardMemoryBuffer.length;

        var address = address(FORWARD_MEMORY_TYPE, forwardMemoryIndex, length);
        forwardMemoryIndex += length;

        return new TensorPointer(address, dimensions, this);
    }

    public @NonNull TensorPointer allocateBackwardMemory(@NonNull Operation operation, int... dimensions) {
        var length = 1;
        for (var dimension : dimensions) {
            length *= dimension;
        }

        if (trackMemoryAllocation) {
            var allocationsSize = allocationsSize(operation.getBackwardMemoryAllocations());

            var allocated = consumedBackwardMemory.computeIfAbsent(operation, (k) -> new long[1]);
            if (length + allocated[0] > allocationsSize) {
                throw new IllegalStateException("Memory allocation exceeded the required memory size for operation "
                        + operation);
            }

            allocated[0] += length;
        }

        assert backwardMemoryIndex + length <= currentStepBackwardMemoryBuffer.length;
        var address = address(currentBackwardMemoryBufferFlag, backwardMemoryIndex, length);

        backwardMemoryIndex += length;

        return new TensorPointer(address, dimensions, this);
    }


    private TensorPointer executeForwardPropagation() {
        return terminalOperation.forwardPassCalculation();
    }

    private void optimizeExecutionGraph() {
        var startOperations = new HashSet<StartOperation>();
        var visitedOperations = new HashSet<Operation>();

        traverseExecutionGraphBackward(terminalOperation, startOperations, visitedOperations);

        visitedOperations.clear();

        for (var startOperation : startOperations) {
            collapseSoftMaxCrossEntropy(startOperation, visitedOperations);
        }
    }

    private void traverseExecutionGraphBackward(Operation operation, HashSet<StartOperation> startOperations,
                                                HashSet<Operation> visitedOperations) {
        if (!visitedOperations.add(operation)) {
            return;
        }

        if (operation instanceof StartOperation startOperation) {
            startOperations.add(startOperation);
        }

        var leftOperation = operation.getLeftPreviousOperation();
        if (leftOperation != null) {
            traverseExecutionGraphBackward(leftOperation, startOperations, visitedOperations);
        }

        var rightOperation = operation.getRightPreviousOperation();
        if (rightOperation != null) {
            traverseExecutionGraphBackward(rightOperation, startOperations, visitedOperations);
        }
    }

    private void collapseSoftMaxCrossEntropy(Operation operation,
                                             HashSet<Operation> visitedOperations) {
        if (!visitedOperations.add(operation)) {
            return;
        }

        var nextTestedOperation = operation.getNextOperation();

        if (operation instanceof SoftMax) {
            if (nextTestedOperation instanceof CrossEntropyCostFunction crossEntropyFunction) {
                var previousOperation = operation.getLeftPreviousOperation();

                var nextOperation = crossEntropyFunction.getNextOperation();
                nextTestedOperation = nextOperation;

                previousOperation.clearNextOperation();
                var expectedValues = crossEntropyFunction.getExpectedValues();
                expectedValues.clearNextOperation();

                var softMaxCrossEntropy = new SoftMaxCrossEntropyCostFunction(expectedValues, previousOperation);

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
        for (var i = layers.size() - 1; i >= 0; i--) {
            backStep(layers.get(i));
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

    private void backStep(ArrayList<Operation> operations) {
        for (var operation : operations) {
            if (operation instanceof StartOperation startOperation) {
                startOperation.calculateGradientUpdate();
            }

            var leftOperation = operation.getLeftPreviousOperation();

            if (leftOperation != null && leftOperation.requiresBackwardDerivativeChainValue()) {
                var result = operation.leftBackwardDerivativeChainValue();
                leftOperation.updateBackwardDerivativeChainValue(result);
            }

            var rightOperation = operation.getRightPreviousOperation();
            if (rightOperation != null && rightOperation.requiresBackwardDerivativeChainValue()) {
                var result = operation.rightBackwardDerivativeChainValue();
                rightOperation.updateBackwardDerivativeChainValue(result);
            }
        }
    }


    private void initializeBuffers() {
        var forwardBufferLength = 0;
        var backwardBufferLength = 0;


        for (var operations : layers) {
            var singleLayerBackwardBufferLength = 0;

            for (var operation : operations) {
                var allocations = operation.getForwardMemoryAllocations();
                forwardBufferLength += allocationsSize(allocations);

                allocations = operation.getBackwardMemoryAllocations();
                singleLayerBackwardBufferLength += allocationsSize(allocations);
            }

            backwardBufferLength = Math.max(singleLayerBackwardBufferLength, backwardBufferLength);
        }

        forwardMemoryBuffer = new float[forwardBufferLength];
        forwardMemoryIndex = 0;

        previousStepBackwardMemoryBuffer = new float[backwardBufferLength];
        currentStepBackwardMemoryBuffer = new float[backwardBufferLength];

        backwardMemoryIndex = 0;
    }

    private static int allocationsSize(int[][] allocations) {
        var sum = 0;

        for (var allocation : allocations) {
            var size = 1;

            for (int j : allocation) {
                size *= j;
            }

            sum += size;
        }

        return sum;
    }

    public static boolean isNull(long address) {
        return address == 0;
    }

    public float @NonNull [] getMemoryBuffer(long address) {
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

    public interface EpochCompletionCallback {
        void onEpochCompleted(int epochIndex, float costResult);
    }
}


