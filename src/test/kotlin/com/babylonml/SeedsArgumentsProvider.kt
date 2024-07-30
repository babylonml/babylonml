package com.babylonml

import org.junit.jupiter.api.extension.ExtensionContext
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.ArgumentsProvider
import java.nio.ByteBuffer
import java.security.SecureRandom
import java.util.stream.Stream

private const val DEFAULT_BATCH_SIZE = 10

class SeedsArgumentsProvider : ArgumentsProvider {
    private val securesRandom = SecureRandom()

    override fun provideArguments(extensionContext: ExtensionContext): Stream<out Arguments> {
        val testMethod = extensionContext.testMethod.get()
        val testMethodParams = testMethod.parameters

        val seeds = ArrayList<Arguments>()
        val batchSize =
            extensionContext.testMethod.get().getAnnotation(SeedBatchSize::class.java)?.value ?: DEFAULT_BATCH_SIZE
        for (k in 0 until batchSize) {
            val seedArgs = Array(testMethodParams.size) {
                val paramType = testMethodParams[it].type
                if (paramType == Long::class.java) {
                    ByteBuffer.wrap(securesRandom.generateSeed(8)).getLong()
                } else if (paramType == String::class.java) {
                    testMethod.name
                } else {
                    throw IllegalArgumentException("Unsupported parameter type: $paramType")
                }
            }

            seeds.add(Arguments.of(*seedArgs))
        }

        return seeds.stream()
    }
}

@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class SeedBatchSize(val value: Int = DEFAULT_BATCH_SIZE)