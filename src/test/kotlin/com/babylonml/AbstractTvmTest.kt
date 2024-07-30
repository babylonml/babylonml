package com.babylonml

import com.babylonml.backend.TvmArray
import com.babylonml.backend.TvmFloatArray
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.TestInfo
import uk.ac.manchester.tornado.api.GridScheduler
import uk.ac.manchester.tornado.api.TaskGraph
import uk.ac.manchester.tornado.api.TornadoExecutionPlan
import uk.ac.manchester.tornado.api.enums.DataTransferMode
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException

abstract class AbstractTvmTest {
    var testName: String = ""

    @BeforeEach
    fun beforeMethod(testInfo: TestInfo) {
        testName = testInfo.testMethod.get().name
    }

    fun taskGraph(vararg inputs: TvmArray): TaskGraph {
        val taskGraph = TaskGraph("testExecutionPass")
        taskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, *inputs)
        return taskGraph
    }

    fun assertExecution(
        taskGraph: TaskGraph, gridScheduler: GridScheduler,
        vararg result: TvmFloatArray, assertions: () -> Unit
    ) {
        taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, *result)
        val immutableTaskGraph = taskGraph.snapshot()
        try {
            TornadoExecutionPlan(immutableTaskGraph).use { executionPlan ->
                executionPlan.withGridScheduler(gridScheduler).execute()
            }
        } catch (e: TornadoExecutionPlanException) {
            throw RuntimeException("Failed to execute the task graph", e)
        }
        assertions()
    }
}