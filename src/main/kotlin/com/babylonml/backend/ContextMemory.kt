package com.babylonml.backend

import com.babylonml.backend.operations.AbstractOperation
import com.babylonml.backend.tensor.common.TensorPointer
import com.babylonml.backend.tensor.common.TensorPointer.MemoryKind
import it.unimi.dsi.fastutil.Function
import it.unimi.dsi.fastutil.ints.IntImmutableList
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray
import java.util.*

class ContextMemory<T : TornadoNativeArray>(
    val memoryBuffer: T,
    private val memoryKind: MemoryKind,
    private val memoryType: TensorPointer.DType,
    private val trackMemoryAllocation: Boolean
) {
    private val consumedMemory = IdentityHashMap<AbstractOperation, LongArray>()
    private var offset = 0

    fun allocate(
        operation: AbstractOperation, dimensions: IntImmutableList,
        expectedAllocations: Function<AbstractOperation, List<IntImmutableList>>
    ): TensorPointer {
        var length = 1
        for (i in 0 until dimensions.size) {
            length *= dimensions.getInt(i)
        }

        if (trackMemoryAllocation) {
            val allocationsSize = allocationsSize(expectedAllocations.apply(operation))
            val allocated =
                consumedMemory.computeIfAbsent(operation) { _: AbstractOperation -> LongArray(1) }

            check(length + allocated[0] <= allocationsSize) {
                ("Memory allocation exceeded the required memory size for operation "
                        + operation)
            }

            allocated[0] += length.toLong()
        }

        check(offset + length <= memoryBuffer.size) {
            ("Memory allocation exceeded the required memory size for operation "
                    + operation)
        }

        val address = offset
        offset += length

        return TensorPointer(address.toLong(), dimensions, memoryType, memoryKind)
    }

    companion object {
        fun allocationsSize(allocations: List<IntImmutableList>): Int {
            var sum = 0

            for (allocation in allocations) {
                var size = 1

                for (i in 0 until allocation.size) {
                    size *= allocation.getInt(i)
                }

                sum += size
            }

            return sum
        }
    }
}
