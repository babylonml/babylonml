package com.tornadoml.cpu

import org.junit.jupiter.api.extension.ExtensionContext
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.ArgumentsProvider
import java.nio.ByteBuffer
import java.security.SecureRandom
import java.util.stream.Stream

class SeedsArgumentsProvider : ArgumentsProvider {
    private val securesRandom = SecureRandom()

    override fun provideArguments(extensionContext: ExtensionContext): Stream<out Arguments> {
        val methodParametersCount = extensionContext.testMethod.get().parameterCount

        val seeds = ArrayList<Arguments>()

        for (k in 0 until 10) {
            val seedArgs = Array(methodParametersCount) {
                ByteBuffer.wrap(securesRandom.generateSeed(8)).getLong()
            }
            seeds.add(Arguments.of(*seedArgs))
        }

        return seeds.stream()
    }
}