package com.babylonml.backend.inference.tornadovm;

import com.babylonml.backend.common.TensorPointer;
import com.babylonml.backend.inference.operations.Operation;
import it.unimi.dsi.fastutil.Function;
import it.unimi.dsi.fastutil.ints.IntImmutableList;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

import java.util.IdentityHashMap;
import java.util.List;

public final class ContextMemory<T extends TornadoNativeArray> {
    private final IdentityHashMap<Operation, long[]> consumedMemory = new IdentityHashMap<>();
    private final T memoryBuffer;
    private int offset;
    private final boolean trackMemoryAllocation;
    private final TensorPointer.MemoryKind memoryKind;
    private final TensorPointer.DType memoryType;

    public ContextMemory(T memoryBuffer, TensorPointer.MemoryKind memoryKind, TensorPointer.DType memoryType,
                         boolean trackMemoryAllocation) {
        this.memoryBuffer = memoryBuffer;
        this.memoryKind = memoryKind;
        this.trackMemoryAllocation = trackMemoryAllocation;
        this.memoryType = memoryType;
    }

    public TensorPointer allocate(Operation operation, IntImmutableList dimensions,
                                  Function<Operation, List<IntImmutableList>> expectedAllocations) {
        var length = 1;
        for (var dimension : dimensions) {
            length *= dimension;
        }

        if (trackMemoryAllocation) {
            var allocationsSize = allocationsSize(expectedAllocations.apply(operation));
            var allocated = consumedMemory.computeIfAbsent(operation, (_) -> new long[1]);

            if (length + allocated[0] > allocationsSize) {
                throw new IllegalStateException("Memory allocation exceeded the required memory size for operation "
                        + operation);
            }

            allocated[0] += length;
        }

        if (offset + length > memoryBuffer.getSize()) {
            throw new IllegalStateException("Memory allocation exceeded the required memory size for operation "
                    + operation);
        }

        var address = offset;
        offset += length;

        return new TensorPointer(address, dimensions, memoryType, memoryKind);
    }

    public T getMemoryBuffer() {
        return memoryBuffer;
    }

    public static int allocationsSize(List<IntImmutableList> allocations) {
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
}
