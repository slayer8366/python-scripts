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
import io.github.slayer8366.faceswap.core.TrackingSwapper
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
        if (intent == null || action !in listOf(ACTION_DOWNLOAD, ACTION_SWAP, ACTION_PREVIEW)) {
            stopSelf(startId)
            return START_NOT_STICKY
        }
        if (worker?.isAlive == true) {
            // Every startForegroundService() must be answered with startForeground().
            startForeground(NOTIFICATION_ID, notification(currentTitle, "Busy", null), currentType)
            return START_NOT_STICKY
        }

        val processing = action != ACTION_DOWNLOAD
        currentType = when {
            processing && Build.VERSION.SDK_INT >= 35 -> ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
            else -> ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        }
        currentTitle = when (action) {
            ACTION_SWAP -> "Swapping faces"
            ACTION_PREVIEW -> "Previewing one frame"
            else -> "Downloading models"
        }
        val title = currentTitle
        startForeground(NOTIFICATION_ID, notification(title, "Starting", null), currentType)
        Jobs.post(JobStatus.Running(title, "Starting", null))

        cancelled = false
        worker = Thread({
            val result = try {
                when (action) {
                    ACTION_SWAP -> swap(intent)
                    ACTION_PREVIEW -> preview(intent)
                    else -> download(intent)
                }
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
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }, "faceswap-worker").apply { start() }
        return START_NOT_STICKY
    }

    /** Android 15+ caps dataSync / mediaProcessing services at 6 h a day. */
    override fun onTimeout(startId: Int, fgsType: Int) {
        cancelled = true
    }

    private fun download(intent: Intent): JobStatus {
        val title = "Downloading models"
        val variant = SwapModel.valueOf(intent.getStringExtra(EXTRA_MODEL) ?: SwapModel.FAST.name)
        ModelStore(this).download(
            variant,
            progress = { done, total ->
                val f = if (total > 0) done.toFloat() / total else null
                report(title, "%d / %d MB".format(done shr 20, total shr 20), f)
            },
            cancelled = { cancelled },
        )
        return JobStatus.Done("Models ready. ${ModelStore.LICENSE_NOTICE}", null)
    }

    /** Settings shared by a full swap and a one-frame preview. */
    private class Request(intent: Intent) {
        val video: Uri = Uri.parse(intent.getStringExtra(EXTRA_VIDEO))
        val source: Uri = Uri.parse(intent.getStringExtra(EXTRA_SOURCE))
        val reference: Uri? = intent.getStringExtra(EXTRA_REFERENCE)?.let(Uri::parse)
        val mode = SwapMode.valueOf(intent.getStringExtra(EXTRA_MODE) ?: SwapMode.ALL.name)
        val threshold = intent.getFloatExtra(EXTRA_THRESHOLD, 0.35f)
        val label = if (intent.getBooleanExtra(EXTRA_LABEL, true)) DEFAULT_LABEL else null
        val previewUs = intent.getLongExtra(EXTRA_PREVIEW_US, 0L).takeIf { it > 0 }
        val frameAtUs = intent.getLongExtra(EXTRA_FRAME_AT_US, 0L)
        val model = SwapModel.valueOf(intent.getStringExtra(EXTRA_MODEL) ?: SwapModel.FAST.name)
        val cropTracking = intent.getBooleanExtra(EXTRA_CROP_TRACKING, false)
    }

    /** What a job needs once the models and face identities are loaded. */
    private class Loaded(val engine: FaceEngine, val latent: FloatArray, val reference: FloatArray?)

    /** Load the models and the face identities for [req]. */
    private fun <T> withEngine(req: Request, title: String, block: (Loaded) -> T): T {
        report(title, "Loading models", null)
        return FaceEngine(ModelStore(this).load(req.model)).use { engine ->
            report(title, "Reading faces", null)
            val src = engine.faceFromStill(decodeImage(this, req.source).toRgbImage(), "source photo")
            val ref = if (req.mode == SwapMode.REFERENCE) {
                val refUri = req.reference ?: throw NoFaceException("Pick a reference photo.")
                engine.faceFromStill(decodeImage(this, refUri).toRgbImage(), "reference photo").embedding
            } else {
                null
            }
            block(Loaded(engine, engine.swapper.latent(src.embedding!!), ref))
        }
    }

    private fun preview(intent: Intent): JobStatus {
        val req = Request(intent)
        val title = "Previewing one frame"
        return withEngine(req, title) { l ->
            report(title, "Swapping", null)
            val frame = VideoTranscoder(this).frameAt(req.video, req.frameAtUs)
            val before = frame.toBitmap()
            val n = l.engine.swapFrame(frame, l.latent, req.mode, l.reference, req.threshold)
            req.label?.let { LabelOverlay(it, frame.width, frame.height).drawOn(frame) }
            val message = when (n) {
                0 -> "No face was swapped in this frame. Try another moment, or check the mode and threshold."
                1 -> "1 face swapped in this frame."
                else -> "$n faces swapped in this frame."
            }
            JobStatus.Preview(before, frame.toBitmap(), message)
        }
    }

    private fun swap(intent: Intent): JobStatus {
        val req = Request(intent)
        val title = "Swapping faces"
        val temp = File(cacheDir, "swap_${System.currentTimeMillis()}.mp4")
        try {
            return withEngine(req, title) { l ->
                // Links faces across frames so reference matches are cached;
                // crop tracking (opt-in) also skips most full-frame detections.
                val swapper = TrackingSwapper(l.engine, l.latent, req.mode, l.reference, req.threshold, req.cropTracking)
                val processor = FrameProcessor(swapper::process)
                val started = SystemClock.elapsedRealtime()
                val stats = VideoTranscoder(this).run(
                    req.video, temp, processor, req.label, req.previewUs,
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
                JobStatus.Done(
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
            else -> return // previews are shown in the app
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
        const val ACTION_PREVIEW = "io.github.slayer8366.faceswap.PREVIEW"
        const val ACTION_CANCEL = "io.github.slayer8366.faceswap.CANCEL"
        const val EXTRA_VIDEO = "video"
        const val EXTRA_SOURCE = "source"
        const val EXTRA_REFERENCE = "reference"
        const val EXTRA_MODE = "mode"
        const val EXTRA_THRESHOLD = "threshold"
        const val EXTRA_LABEL = "label"
        const val EXTRA_PREVIEW_US = "preview_us"
        const val EXTRA_FRAME_AT_US = "frame_at_us"
        const val EXTRA_MODEL = "model"
        const val EXTRA_CROP_TRACKING = "crop_tracking"
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
