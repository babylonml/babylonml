plugins {
    java
    kotlin("jvm") version "2.0.0"
}

group = "com.tornadoml"
version = "1.0-SNAPSHOT"


java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

repositories {
    mavenCentral()
}

dependencies {
    // Existing dependencies
    implementation("org.apache.commons:commons-rng-core:1.5")
    implementation("org.apache.commons:commons-rng-simple:1.5")
    implementation("org.apache.commons:commons-rng-sampling:1.5")

    // Test dependencies
    testImplementation(platform("org.junit:junit-bom:5.10.2"))
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.withType<JavaCompile> {
    // Add module for Vector API if available and necessary
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}

tasks {
    test {
        // Use JUnit Platform and enable preview features in tests
        useJUnitPlatform()
        jvmArgs = listOf("-Xmx8g", "--add-modules", "jdk.incubator.vector")
    }

    register<JavaExec>("mnistBench") {
        group = "application"
        mainClass = "com.tornadoml.mnist.MNISTBench"
        classpath = sourceSets["main"].runtimeClasspath
        jvmArgs = listOf(
            "-server",
            "-Xmx16g",
            "-XX:MaxDirectMemorySize=110g",
            "-XX:+HeapDumpOnOutOfMemoryError",
            "--add-modules",
            "jdk.incubator.vector",
            "-Djava.awt.headless=true"
        )
    }

    register<JavaExec>("mnistBenchLight") {
        group = "application"
        mainClass = "com.tornadoml.mnist.MNISTBenchLight"
        classpath = sourceSets["main"].runtimeClasspath
        jvmArgs = listOf(
            "-server",
            "-Xmx16g",
            "-XX:MaxDirectMemorySize=110g",
            "-XX:+HeapDumpOnOutOfMemoryError",
            "--add-modules",
            "jdk.incubator.vector",
            "-Djava.awt.headless=true"
        )
    }
}



