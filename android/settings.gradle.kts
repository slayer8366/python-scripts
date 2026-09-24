pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    plugins {
        id("com.android.application") version "8.13.0"
        id("org.jetbrains.kotlin.android") version "2.2.20"
        id("org.jetbrains.kotlin.jvm") version "2.2.20"
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "FaceSwapVideo"
include(":core")
// `-PcoreOnly` builds and tests the pure-Kotlin core without the Android
// Gradle plugin, for machines that can't reach Google's Maven repository.
if (!providers.gradleProperty("coreOnly").isPresent) {
    include(":app")
}
