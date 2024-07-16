package com.babylonml

import com.babylonml.backend.inference.operations.TvmArray
import com.babylonml.backend.inference.operations.TvmFloatArray
import uk.ac.manchester.tornado.api.TaskGraph
import uk.ac.manchester.tornado.api.TornadoExecutionPlan
import uk.ac.manchester.tornado.api.enums.DataTransferMode
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException

abstract class AbstractTvmTest {
    fun taskGraph(vararg inputs: TvmArray): TaskGraph {
        val taskGraph = TaskGraph("executionPass")
        taskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, *inputs)
        return taskGraph
    }

    fun assertExecution(taskGraph: TaskGraph, vararg result: TvmFloatArray, assertions: () -> Unit) {
        taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, *result)
        val immutableTaskGraph = taskGraph.snapshot()
        try {
            TornadoExecutionPlan(immutableTaskGraph).use { executionPlan ->
                executionPlan.execute()
            }
        } catch (e: TornadoExecutionPlanException) {
            throw RuntimeException("Failed to execute the task graph", e)
        }
        assertions()
    }


}