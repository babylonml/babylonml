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
        val methodParametersCount = extensionContext.testMethod.get().parameterCount

        val seeds = ArrayList<Arguments>()
        val batchSize =
            extensionContext.testMethod.get().getAnnotation(SeedBatchSize::class.java)?.value ?: DEFAULT_BATCH_SIZE
        for (k in 0 until batchSize) {
            val seedArgs = Array(methodParametersCount) {
                ByteBuffer.wrap(securesRandom.generateSeed(8)).getLong()
            }
            seeds.add(Arguments.of(*seedArgs))
        }

        return seeds.stream()
    }
}

@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class SeedBatchSize(val value: Int = DEFAULT_BATCH_SIZE)