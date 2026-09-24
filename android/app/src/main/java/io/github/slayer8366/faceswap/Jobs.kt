package io.github.slayer8366.faceswap

import android.net.Uri
import android.os.Handler
import android.os.Looper

/** What the background service is doing, observed by the UI on the main thread. */
sealed interface JobStatus {
    data object Idle : JobStatus
    data class Running(val title: String, val detail: String, val fraction: Float?) : JobStatus
    data class Done(val message: String, val output: Uri?) : JobStatus
    data class Failed(val message: String) : JobStatus
}

object Jobs {
    private val main = Handler(Looper.getMainLooper())
    private val listeners = LinkedHashSet<(JobStatus) -> Unit>()

    @Volatile
    var status: JobStatus = JobStatus.Idle
        private set

    val isRunning get() = status is JobStatus.Running

    fun post(s: JobStatus) {
        status = s
        main.post { listeners.toList().forEach { it(s) } }
    }

    /** Main thread only. Delivers the current status immediately. */
    fun observe(listener: (JobStatus) -> Unit) {
        listeners.add(listener)
        listener(status)
    }

    fun unobserve(listener: (JobStatus) -> Unit) {
        listeners.remove(listener)
    }
}
