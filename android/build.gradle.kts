// Load the Kotlin plugins once for all subprojects (Gradle warns otherwise).
// The Android plugin isn't listed so `-PcoreOnly` builds work without Google's Maven.
plugins {
    id("org.jetbrains.kotlin.jvm") apply false
    id("org.jetbrains.kotlin.android") apply false
}
