package com.babylonml.dsl.v1

object MatrixDemo {
    @JvmStatic
    fun main(args: Array<String>) {

        println("Vector dot product")
        val row1 = Matrix.rowVector(1.0f, 2.0f, 3.0f)
        val col1 = Matrix.columnVector(1.0f, 2.0f, 3.0f)

        val result1 = row1 * col1
        println(result1.materialize().contentToString())


        println("Broadcasting example")
        val m2x3 = Matrix.of(
            arrayOf(
                floatArrayOf(1.0f, 2.0f, 3.0f),
                floatArrayOf(4.0f, 5.0f, 6.0f)
            )
        )

        val v1x3 = Matrix.rowVector(0.5f, 0.6f, 0.7f)
        val result3 = m2x3 + v1x3
        println(result3.materialize().contentToString())


        println("Broadcasting example 2")

        val v2x1 = Matrix.columnVector(0.8f, 0.9f)
        val result4 = m2x3 + v2x1
        println(result4.materialize().contentToString())



        println("Broadcasting example 3")
        val v4x1 = Matrix.columnVector(11f, 12f, 13f, 14f)
        val v1x5 = Matrix.rowVector(0.1f, 0.2f, 0.3f, 0.4f, 0.5f)
        val result6 = v4x1 + v1x5
        println(result6.materialize().contentToString())
    }
}
