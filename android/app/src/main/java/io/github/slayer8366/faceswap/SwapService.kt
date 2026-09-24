package io.github.slayer8366.faceswap

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import android.provider.MediaStore
import io.github.slayer8366.faceswap.core.FaceEngine
import io.github.slayer8366.faceswap.core.NoFaceException
import io.github.slayer8366.faceswap.core.SwapMode
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.CancellationException
import kotlin.math.roundToInt

/** Runs model downloads and video swaps in the foreground, one at a time. */
class SwapService : Service() {
    @Volatile private var cancelled = false
    private var worker: Thread? = null
    private var lastNotify = 0L
    private var currentType = ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
    private var currentTitle = ""

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action
        if (action == ACTION_CANCEL) {
            cancelled = true
            return START_NOT_STICKY
        }
        if (intent == null || (action != ACTION_DOWNLOAD && action != ACTION_SWAP)) {
            stopSelf(startId)
            return START_NOT_STICKY
        }
        if (worker?.isAlive == true) {
            // Every startForegroundService() must be answered with startForeground().
            startForeground(NOTIFICATION_ID, notification(currentTitle, "Busy", null), currentType)
            return START_NOT_STICKY
        }

        val swapping = action == ACTION_SWAP
        currentType = when {
            swapping && Build.VERSION.SDK_INT >= 35 -> ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
            else -> ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        }
        currentTitle = if (swapping) "Swapping faces" else "Downloading models"
        val title = currentTitle
        startForeground(NOTIFICATION_ID, notification(title, "Starting", null), currentType)
        Jobs.post(JobStatus.Running(title, "Starting", null))

        cancelled = false
        worker = Thread({
            val result = try {
                if (swapping) swap(intent) else download()
            } catch (e: CancellationException) {
                JobStatus.Failed("Cancelled.")
            } catch (e: InterruptedException) {
                JobStatus.Failed("Cancelled.")
            } catch (e: OutOfMemoryError) {
                JobStatus.Failed("Ran out of memory. The models need about 1 GB free; close other apps and retry.")
            } catch (e: Exception) {
                JobStatus.Failed(e.message ?: e.javaClass.simpleName)
            }
            Jobs.post(result)
            finishNotification(result)
            stopForeground(STOP_FOREGROUND_DETACH)
            stopSelf()
        }, "faceswap-worker").apply { start() }
        return START_NOT_STICKY
    }

    /** Android 15+ caps dataSync / mediaProcessing services at 6 h a day. */
    override fun onTimeout(startId: Int, fgsType: Int) {
        cancelled = true
    }

    private fun download(): JobStatus {
        val title = "Downloading models"
        ModelStore(this).download(
            progress = { done, total ->
                val f = if (total > 0) done.toFloat() / total else null
                report(title, "%d / %d MB".format(done shr 20, total shr 20), f)
            },
            cancelled = { cancelled },
        )
        return JobStatus.Done("Models ready. ${ModelStore.LICENSE_NOTICE}", null)
    }

    private fun swap(intent: Intent): JobStatus {
        val title = "Swapping faces"
        val video = Uri.parse(intent.getStringExtra(EXTRA_VIDEO))
        val source = Uri.parse(intent.getStringExtra(EXTRA_SOURCE))
        val reference = intent.getStringExtra(EXTRA_REFERENCE)?.let(Uri::parse)
        val mode = SwapMode.valueOf(intent.getStringExtra(EXTRA_MODE) ?: SwapMode.ALL.name)
        val threshold = intent.getFloatExtra(EXTRA_THRESHOLD, 0.35f)
        val label = if (intent.getBooleanExtra(EXTRA_LABEL, true)) DEFAULT_LABEL else null
        val previewUs = intent.getLongExtra(EXTRA_PREVIEW_US, 0L).takeIf { it > 0 }

        report(title, "Loading models", null)
        val temp = File(cacheDir, "swap_${System.currentTimeMillis()}.mp4")
        try {
            FaceEngine(ModelStore(this).load()).use { engine ->
                report(title, "Reading faces", null)
                val src = engine.faceFromStill(decodeImage(this, source).toRgbImage(), "source photo")
                val ref = if (mode == SwapMode.REFERENCE) {
                    engine.faceFromStill(decodeImage(this, reference ?: throw NoFaceException("Pick a reference photo.")).toRgbImage(), "reference photo").embedding
                } else {
                    null
                }
                val settings = SwapSettings(engine.swapper.latent(src.embedding!!), mode, ref, threshold, label, previewUs)
                val started = SystemClock.elapsedRealtime()
                val stats = VideoSwapper(this, engine).run(
                    video, temp, settings,
                    progress = { f, s ->
                        val secs = (SystemClock.elapsedRealtime() - started) / 1000.0
                        val eta = if (f > 0.01f) " · ~${((secs / f - secs) / 60).roundToInt()} min left" else ""
                        report(title, "${s.frames} frames, ${s.faces} faces swapped$eta", f)
                    },
                    cancelled = { cancelled },
                )
                report(title, "Saving to gallery", null)
                val uri = saveToGallery(temp)
                val audio = when (stats.audio) {
                    AudioResult.COPIED -> ""
                    AudioResult.NONE -> " No audio track."
                    AudioResult.UNSUPPORTED -> " Audio format couldn't be copied; output is silent."
                }
                return JobStatus.Done(
                    "Saved to Movies/FaceSwap. ${stats.frames} frames, ${stats.faces} faces swapped " +
                        "in ${stats.framesWithSwap} frames.$audio",
                    uri,
                )
            }
        } finally {
            temp.delete()
        }
    }

    private fun saveToGallery(file: File): Uri {
        val name = "faceswap_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".mp4"
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, name)
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/FaceSwap")
            put(MediaStore.Video.Media.DESCRIPTION, "AI-generated face swap")
            put(MediaStore.Video.Media.IS_PENDING, 1)
        }
        val resolver = contentResolver
        val uri = resolver.insert(MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values)
            ?: throw IllegalStateException("Could not create a gallery entry")
        try {
            resolver.openOutputStream(uri)!!.use { out -> file.inputStream().use { it.copyTo(out) } }
            resolver.update(uri, ContentValues().apply { put(MediaStore.Video.Media.IS_PENDING, 0) }, null, null)
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            throw e
        }
        return uri
    }

    private fun report(title: String, detail: String, fraction: Float?) {
        Jobs.post(JobStatus.Running(title, detail, fraction))
        val now = SystemClock.elapsedRealtime()
        if (now - lastNotify > 1000) {
            lastNotify = now
            getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification(title, detail, fraction))
        }
    }

    private fun finishNotification(status: JobStatus) {
        val text = when (status) {
            is JobStatus.Done -> status.message
            is JobStatus.Failed -> status.message
            else -> return
        }
        val done = Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download_done)
            .setContentTitle(if (status is JobStatus.Done) "Finished" else "Stopped")
            .setContentText(text)
            .setStyle(Notification.BigTextStyle().bigText(text))
            .setContentIntent(openApp())
            .setAutoCancel(true)
            .build()
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID + 1, done)
    }

    private fun notification(title: String, detail: String, fraction: Float?): Notification {
        ensureChannel(this)
        val cancel = PendingIntent.getService(
            this, 1, Intent(this, SwapService::class.java).setAction(ACTION_CANCEL),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle(title)
            .setContentText(detail)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(openApp())
            .setProgress(1000, ((fraction ?: 0f) * 1000).roundToInt(), fraction == null)
            .addAction(Notification.Action.Builder(null, "Cancel", cancel).build())
            .build()
    }

    private fun openApp(): PendingIntent = PendingIntent.getActivity(
        this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
        PendingIntent.FLAG_IMMUTABLE,
    )

    companion object {
        const val ACTION_DOWNLOAD = "io.github.slayer8366.faceswap.DOWNLOAD"
        const val ACTION_SWAP = "io.github.slayer8366.faceswap.SWAP"
        const val ACTION_CANCEL = "io.github.slayer8366.faceswap.CANCEL"
        const val EXTRA_VIDEO = "video"
        const val EXTRA_SOURCE = "source"
        const val EXTRA_REFERENCE = "reference"
        const val EXTRA_MODE = "mode"
        const val EXTRA_THRESHOLD = "threshold"
        const val EXTRA_LABEL = "label"
        const val EXTRA_PREVIEW_US = "preview_us"
        private const val CHANNEL_ID = "jobs"
        private const val NOTIFICATION_ID = 1

        fun ensureChannel(context: Context) {
            val nm = context.getSystemService(NotificationManager::class.java)
            if (nm.getNotificationChannel(CHANNEL_ID) == null) {
                nm.createNotificationChannel(NotificationChannel(CHANNEL_ID, "Face swap jobs", NotificationManager.IMPORTANCE_LOW))
            }
        }
    }
}
