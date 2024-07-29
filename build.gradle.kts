import net.ltgt.gradle.errorprone.CheckSeverity
import net.ltgt.gradle.errorprone.errorprone
import java.nio.file.Files
import java.nio.file.Paths
import java.util.Properties
import kotlin.io.path.absolute
import kotlin.io.path.inputStream

val TORNADOVM_PROVIDERS = listOf(
    "-Dtornado.load.api.implementation=uk.ac.manchester.tornado.runtime.tasks.TornadoTaskGraph",
    "-Dtornado.load.runtime.implementation=uk.ac.manchester.tornado.runtime.TornadoCoreRuntime",
    "-Dtornado.load.tornado.implementation=uk.ac.manchester.tornado.runtime.common.Tornado",
    "-Dtornado.load.annotation.implementation=uk.ac.manchester.tornado.annotation.ASMClassVisitor",
    "-Dtornado.load.annotation.parallel=uk.ac.manchester.tornado.api.annotations.Parallel"
)

val TVM_COMMON_EXPORTS = "etc/exportLists/common-exports"
val TVM_OPENCL_EXPORTS = "etc/exportLists/opencl-exports"
val TVM_PTX_EXPORTS = "etc/exportLists/ptx-exports"
val TVM_SPIRV_EXPORTS = "etc/exportLists/spirv-exports"
val TVM_MODULES = "ALL-SYSTEM,tornado.runtime,tornado.annotation,tornado.drivers.common"
val TVM_PTX_MODULE = "tornado.drivers.ptx"
val TVM_OPENCL_MODULE = "tornado.drivers.opencl"

val TVM_JAVA_BASE_OPTIONS = listOf(
    "-server", "-XX:+UnlockExperimentalVMOptions", "-XX:+EnableJVMCI", "--enable-preview"
)

val TVM_JAVA_GC_JDK16 = listOf("-XX:+UseParallelGC")

val tvmArgs = createTornadoVMArgs()

plugins {
    java
    kotlin("jvm") version "2.0.0"
    id("net.ltgt.errorprone") version "4.0.1"
}

group = "com.babylonml"
version = "1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

repositories {
    mavenCentral()
    mavenLocal()

    maven {
        url = uri("https://raw.githubusercontent.com/beehive-lab/tornado/maven-tornadovm")
    }
}

dependencies {
    implementation("tornado:tornado-api:1.0.6")
    implementation("org.apache.commons:commons-rng-core:1.6")
    implementation("org.apache.commons:commons-rng-simple:1.5")
    implementation("org.apache.commons:commons-rng-sampling:1.5")
    implementation("net.openhft:zero-allocation-hashing:0.26ea0")
    implementation("it.unimi.dsi:fastutil:8.5.14")
    implementation("org.jspecify:jspecify:0.3.0")

    // Test dependencies
    testImplementation(platform("org.junit:junit-bom:5.10.3"))
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testImplementation("org.junit.jupiter:junit-jupiter")

    errorprone("com.google.errorprone:error_prone_core:2.29.2")
    errorprone("com.uber.nullaway:nullaway:0.11.0")
}

tasks.withType<JavaCompile> {
    // Add module for Vector API if available and necessary
    options.compilerArgs.addAll(listOf("--add-modules", "ALL-SYSTEM", "--enable-preview"))

    if (!name.lowercase().contains("test")) {
        options.errorprone {
            check("NullAway", CheckSeverity.ERROR)
            option("NullAway:AnnotatedPackages", "com.babylonml")
        }
    }
}

tasks {
    test {
        // Use JUnit Platform and enable preview features in tests
        useJUnitPlatform()
        if (tvmArgs == null) {
            error("TornadoVM args could not be created")
        }

        jvmArgs = tvmArgs + listOf("-Xmx8g", "-da:org.graalvm.compiler...")
    }

    register<JavaExec>("mnistBench") {
        group = "application"
        mainClass = "com.babylonml.backend.examples.mnist.MNISTBench"
        classpath = sourceSets["main"].runtimeClasspath
        jvmArgs = listOf(
            "-Xmx16g",
        )
    }

    register<JavaExec>("mnistBenchLight") {
        group = "application"
        mainClass = "com.babylonml.backend.examples.mnist.MNISTBenchLight"
        classpath = sourceSets["main"].runtimeClasspath
        jvmArgs = listOf(
            "-server",
            "-Xmx16g",
            "--add-modules",
            "jdk.incubator.vector",
            "--enable-preview",
            "-Djava.awt.headless=true"
        )
    }


    register<JavaExec>("makemore") {
        group = "application"
        mainClass = "com.tornadoml.makemore.Makemore"
        classpath = sourceSets["main"].runtimeClasspath
        jvmArgs = listOf(
            "-server",
            "-Xmx16g",
            "--add-modules",
            "jdk.incubator.vector",
            "--enable-preview",
        )
    }

    register<JavaExec>("showDevices") {
        val tvmArgs = createTornadoVMArgs() ?: error("TornadoVM args could not be created")
        jvmArgs = tvmArgs + listOf("-Xmx16g")
        args = listOf("verbose")
        group = "application"
        mainClass = "uk.ac.manchester.tornado.drivers.TornadoDeviceQuery"
        classpath = sourceSets["main"].runtimeClasspath

    }
}

fun createTornadoVMArgs(): List<String>? {
    val tvmHomeString = providers.gradleProperty("babylonml.tornadovm.home").orNull
    if (tvmHomeString == null) {
        logger.error("babylonml.tornadovm.home property must be set")
        return null
    }

    val tvmHome = Paths.get(tvmHomeString)
    logger.info("TornadoVM Home: ${tvmHome.absolute()}")

    val tornadoBackendFilePath = tvmHome.resolve("etc/tornado.backend")
    if (!Files.exists(tornadoBackendFilePath)) {
        logger.error("TornadoVM backend file not found: $tornadoBackendFilePath")
        return null
    }

    val tornadoBackends = Properties().apply {
        load(tornadoBackendFilePath.inputStream())
    }

    val backendsString = tornadoBackends.getProperty("tornado.backends")
    if (backendsString == null) {
        logger.error("TornadoVM backend file does not contain tornado.backends property")
        return null
    }

    val backendsList = backendsString.split(",")
    logger.info("Tornado backends: $backendsList")

    var tvmModules = TVM_MODULES
    val upgradeModulePath = tvmHome.resolve("share/java/graalJars").absolute().toString()

    val common = tvmHome.resolve(TVM_COMMON_EXPORTS).absolute().toString()
    val opencl = tvmHome.resolve(TVM_OPENCL_EXPORTS).absolute().toString()
    val ptx = tvmHome.resolve(TVM_PTX_EXPORTS).absolute().toString()
    val spirv = tvmHome.resolve(TVM_SPIRV_EXPORTS).absolute().toString()

    val exports = mutableListOf("@$common")
    if (backendsList.contains("opencl-backend")) {
        tvmModules += ",$TVM_OPENCL_MODULE"
        exports.add("@$opencl")
    }
    if (backendsList.contains("ptx-backend")) {
        tvmModules += ",$TVM_PTX_MODULE"
        exports.add("@$ptx")
    }
    if (backendsList.contains("spirv-backend")) {
        exports.add("@$spirv")
        exports.add("@$opencl")

        tvmModules += ",$TVM_OPENCL_MODULE"
    }

    val modulePath = "." + File.pathSeparator + tvmHome.resolve("share/java/tornado").absolute().toString()
    val libraryPath = tvmHome.resolve("lib").absolute().toString()

    logger.info("Upgrade module path: $upgradeModulePath")
    logger.info("TVM modules: $tvmModules")
    logger.info("Module path: $modulePath")
    logger.info("Exports: $exports")
    logger.info("Library path: $libraryPath")

    val result = TVM_JAVA_BASE_OPTIONS + TVM_JAVA_GC_JDK16 + listOf(
        "-Djava.library.path=$libraryPath",
        "--module-path",
        modulePath,
        "--add-modules",
        tvmModules,
        "--upgrade-module-path",
        upgradeModulePath,
    ) + exports + TORNADOVM_PROVIDERS

    return result
}
