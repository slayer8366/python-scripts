import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("org.jetbrains.kotlin.jvm")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions.jvmTarget.set(JvmTarget.JVM_17)
}

dependencies {
    // The ai.onnxruntime API is identical in the desktop and Android
    // artifacts; the app supplies onnxruntime-android at runtime.
    compileOnly("com.microsoft.onnxruntime:onnxruntime:1.30.0")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.30.0")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
    maxHeapSize = "4g"
    // Model-backed tests run only when these point at real files.
    listOf("FACESWAP_MODEL_DIR", "FACESWAP_GOLDEN_DIR").forEach { name ->
        System.getenv(name)?.let { environment(name, it) }
    }
    testLogging { events("passed", "skipped", "failed"); showStandardStreams = true }
}
