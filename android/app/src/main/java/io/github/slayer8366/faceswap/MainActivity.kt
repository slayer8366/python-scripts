package io.github.slayer8366.faceswap

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowInsets
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import io.github.slayer8366.faceswap.core.SwapMode
import kotlin.math.roundToInt

class MainActivity : Activity() {
    private var sourceUri: Uri? = null
    private var videoUri: Uri? = null
    private var referenceUri: Uri? = null
    private var resultUri: Uri? = null

    private lateinit var models: ModelStore
    private lateinit var modelStatus: TextView
    private lateinit var downloadButton: Button
    private lateinit var sourceImage: ImageView
    private lateinit var videoName: TextView
    private lateinit var modeGroup: RadioGroup
    private lateinit var referenceSection: LinearLayout
    private lateinit var referenceImage: ImageView
    private lateinit var thresholdLabel: TextView
    private lateinit var threshold: SeekBar
    private lateinit var labelBox: CheckBox
    private lateinit var previewBox: CheckBox
    private lateinit var consentBox: CheckBox
    private lateinit var swapButton: Button
    private lateinit var progress: ProgressBar
    private lateinit var statusText: TextView
    private lateinit var cancelButton: Button
    private lateinit var resultRow: LinearLayout
    private lateinit var previewAtLabel: TextView
    private lateinit var previewAt: SeekBar
    private lateinit var previewButton: Button
    private lateinit var previewRow: LinearLayout
    private lateinit var previewBefore: ImageView
    private lateinit var previewAfter: ImageView
    private var videoDurationMs = 0L

    private val statusListener: (JobStatus) -> Unit = { render(it) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        models = ModelStore(this)
        setContentView(buildUi())
        savedInstanceState?.let { s ->
            sourceUri = s.getString(KEY_SOURCE)?.let(Uri::parse)
            videoUri = s.getString(KEY_VIDEO)?.let(Uri::parse)
            referenceUri = s.getString(KEY_REFERENCE)?.let(Uri::parse)
        }
        sourceUri?.let { showThumb(sourceImage, it) }
        referenceUri?.let { showThumb(referenceImage, it) }
        videoUri?.let(::onVideoChosen)
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 0)
        }
    }

    override fun onStart() {
        super.onStart()
        Jobs.observe(statusListener)
    }

    override fun onStop() {
        Jobs.unobserve(statusListener)
        super.onStop()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        sourceUri?.let { outState.putString(KEY_SOURCE, it.toString()) }
        videoUri?.let { outState.putString(KEY_VIDEO, it.toString()) }
        referenceUri?.let { outState.putString(KEY_REFERENCE, it.toString()) }
    }

    @Deprecated("Framework Activity result API; no AndroidX in this app.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data
        if (resultCode != RESULT_OK || uri == null) return
        when (requestCode) {
            PICK_SOURCE -> { sourceUri = uri; showThumb(sourceImage, uri) }
            PICK_REFERENCE -> { referenceUri = uri; showThumb(referenceImage, uri) }
            PICK_VIDEO -> { videoUri = uri; onVideoChosen(uri) }
        }
        updateEnabled()
    }

    private fun pick(mime: String, request: Int) {
        val intent = if (Build.VERSION.SDK_INT >= 33) {
            Intent(MediaStore.ACTION_PICK_IMAGES).setType(mime)
        } else {
            Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType(mime)
        }
        @Suppress("DEPRECATION")
        startActivityForResult(intent, request)
    }

    private fun selectedMode() = when (modeGroup.checkedRadioButtonId) {
        ID_LARGEST -> SwapMode.LARGEST
        ID_REFERENCE -> SwapMode.REFERENCE
        else -> SwapMode.ALL
    }

    private fun thresholdValue() = threshold.progress / 100f

    private fun jobIntent(action: String): Intent {
        val mode = selectedMode()
        val intent = Intent(this, SwapService::class.java).setAction(action)
            .putExtra(SwapService.EXTRA_SOURCE, sourceUri.toString())
            .putExtra(SwapService.EXTRA_VIDEO, videoUri.toString())
            .putExtra(SwapService.EXTRA_MODE, mode.name)
            .putExtra(SwapService.EXTRA_THRESHOLD, thresholdValue())
            .putExtra(SwapService.EXTRA_LABEL, labelBox.isChecked)
            .putExtra(SwapService.EXTRA_PREVIEW_US, if (previewBox.isChecked) PREVIEW_US else 0L)
            .putExtra(SwapService.EXTRA_FRAME_AT_US, previewAt.progress * 100_000L)
        if (mode == SwapMode.REFERENCE) intent.putExtra(SwapService.EXTRA_REFERENCE, referenceUri.toString())
        return intent
    }

    private fun onVideoChosen(uri: Uri) {
        videoName.text = displayName(uri)
        videoDurationMs = try {
            MediaMetadataRetriever().run {
                try {
                    setDataSource(this@MainActivity, uri)
                    extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                } finally {
                    release()
                }
            }
        } catch (e: Exception) {
            0L
        }
        // Slider steps are 0.1 s; start a third of the way in, where faces are likelier than on frame one.
        previewAt.max = (videoDurationMs / 100).toInt().coerceAtLeast(0)
        previewAt.progress = previewAt.max / 3
        updatePreviewAtLabel()
    }

    private fun updatePreviewAtLabel() {
        val tenths = previewAt.progress
        previewAtLabel.text = "Preview frame at %d:%02d.%d".format(tenths / 600, tenths / 10 % 60, tenths % 10)
    }

    private fun render(status: JobStatus) {
        val running = status is JobStatus.Running
        progress.visibility = if (running) View.VISIBLE else View.GONE
        cancelButton.visibility = if (running) View.VISIBLE else View.GONE
        when (status) {
            is JobStatus.Idle -> statusText.text = ""
            is JobStatus.Running -> {
                statusText.text = "${status.title}: ${status.detail}"
                progress.isIndeterminate = status.fraction == null
                status.fraction?.let { progress.progress = (it * 1000).roundToInt() }
            }
            is JobStatus.Done -> {
                statusText.text = status.message
                resultUri = status.output
            }
            is JobStatus.Failed -> statusText.text = "Stopped: ${status.message}"
            is JobStatus.Preview -> {
                statusText.text = status.message
                previewBefore.setImageBitmap(status.before)
                previewAfter.setImageBitmap(status.after)
            }
        }
        previewRow.visibility = if (status is JobStatus.Preview) View.VISIBLE else View.GONE
        resultRow.visibility = if (!running && resultUri != null) View.VISIBLE else View.GONE
        refreshModels()
        updateEnabled()
    }

    private fun refreshModels() {
        val ready = models.isReady
        modelStatus.text = if (ready) {
            "Models installed. ${ModelStore.LICENSE_NOTICE}"
        } else {
            "The face models (about 840 MB) must be downloaded once before swapping. Use Wi-Fi. " +
                ModelStore.LICENSE_NOTICE
        }
        downloadButton.visibility = if (ready) View.GONE else View.VISIBLE
    }

    private fun updateEnabled() {
        val running = Jobs.isRunning
        val needsReference = selectedMode() == SwapMode.REFERENCE
        downloadButton.isEnabled = !running
        val ready = !running && models.isReady && consentBox.isChecked &&
            sourceUri != null && videoUri != null && (!needsReference || referenceUri != null)
        swapButton.isEnabled = ready
        previewButton.isEnabled = ready
        val hasVideo = videoUri != null && videoDurationMs > 0
        previewAtLabel.visibility = if (hasVideo) View.VISIBLE else View.GONE
        previewAt.visibility = if (hasVideo) View.VISIBLE else View.GONE
        referenceSection.visibility = if (needsReference) View.VISIBLE else View.GONE
    }

    private fun showThumb(view: ImageView, uri: Uri) {
        try {
            view.setImageBitmap(decodeImage(this, uri, maxSide = 512))
            view.visibility = View.VISIBLE
        } catch (e: Exception) {
            Toast.makeText(this, "Couldn't open that image: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun displayName(uri: Uri): String =
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        } ?: uri.lastPathSegment ?: "video"

    private fun openResult(action: String) {
        val uri = resultUri ?: return
        val intent = if (action == Intent.ACTION_SEND) {
            Intent(Intent.ACTION_SEND).setType("video/mp4").putExtra(Intent.EXTRA_STREAM, uri)
        } else {
            Intent(Intent.ACTION_VIEW).setDataAndType(uri, "video/mp4")
        }
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        startActivity(Intent.createChooser(intent, null))
    }

    // --- Layout -----------------------------------------------------------

    private fun dp(v: Int) = (v * resources.displayMetrics.density).roundToInt()

    private fun text(s: String, size: Float = 15f, bold: Boolean = false) = TextView(this).apply {
        text = s
        textSize = size
        if (bold) setTypeface(typeface, Typeface.BOLD)
        setPadding(0, dp(6), 0, dp(6))
    }

    private fun button(label: String, onClick: () -> Unit) = Button(this).apply {
        text = label
        setOnClickListener { onClick() }
    }

    private fun thumb() = ImageView(this).apply {
        layoutParams = LinearLayout.LayoutParams(dp(120), dp(120))
        scaleType = ImageView.ScaleType.CENTER_CROP
        visibility = View.GONE
    }

    private fun buildUi(): View {
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(24))
        }
        fun add(v: View) = column.addView(v)
        fun section(title: String) = add(text(title, 17f, bold = true).apply { setPadding(0, dp(16), 0, dp(4)) })

        add(text("Put the face from one photo onto faces in a video, entirely on this phone.", 15f))
        add(text("Only use faces of people who have agreed to it. Output is labelled AI-generated by default.", 13f))

        section("1. Models")
        modelStatus = text("", 13f).also(::add)
        downloadButton = button("Download models") {
            startForegroundService(Intent(this, SwapService::class.java).setAction(SwapService.ACTION_DOWNLOAD))
        }.also(::add)

        section("2. Face to insert")
        add(button("Choose photo") { pick("image/*", PICK_SOURCE) })
        sourceImage = thumb().also(::add)

        section("3. Video")
        add(button("Choose video") { pick("video/*", PICK_VIDEO) })
        videoName = text("", 13f).also(::add)

        section("4. Which faces to replace")
        modeGroup = RadioGroup(this).apply {
            addView(RadioButton(context).apply { id = ID_ALL; text = "Every face" })
            addView(RadioButton(context).apply { id = ID_LARGEST; text = "Largest face in each frame" })
            addView(RadioButton(context).apply { id = ID_REFERENCE; text = "One person (match a photo of them)" })
            check(ID_ALL)
            setOnCheckedChangeListener { _, _ -> updateEnabled() }
        }.also(::add)
        referenceSection = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        referenceSection.addView(button("Choose photo of the person to replace") { pick("image/*", PICK_REFERENCE) })
        referenceImage = thumb().also(referenceSection::addView)
        thresholdLabel = text("", 13f).also(referenceSection::addView)
        threshold = SeekBar(this).apply {
            max = 90
            min = 10
            progress = 35
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar, p: Int, fromUser: Boolean) = updateThresholdLabel()
                override fun onStartTrackingTouch(s: SeekBar) = Unit
                override fun onStopTrackingTouch(s: SeekBar) = Unit
            })
        }.also(referenceSection::addView)
        referenceSection.addView(text("Raise it if the wrong people get swapped; lower it if the person is missed.", 12f))
        add(referenceSection)
        updateThresholdLabel()

        section("5. Options")
        labelBox = CheckBox(this).apply { text = "Burn in \"AI-GENERATED\" label"; isChecked = true }.also(::add)
        previewBox = CheckBox(this).apply { text = "Only process the first 5 seconds"; isChecked = true }.also(::add)
        consentBox = CheckBox(this).apply {
            text = "Everyone whose face is used or replaced has agreed to it"
            setOnCheckedChangeListener { _, _ -> updateEnabled() }
        }.also(::add)
        add(text("Expect several seconds per frame per face on a phone.", 12f))

        section("6. Check one frame first")
        previewAtLabel = text("", 13f).also(::add)
        previewAt = SeekBar(this).apply {
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar, p: Int, fromUser: Boolean) = updatePreviewAtLabel()
                override fun onStartTrackingTouch(s: SeekBar) = Unit
                override fun onStopTrackingTouch(s: SeekBar) = Unit
            })
        }.also(::add)
        previewButton = button("Preview one frame") {
            startForegroundService(jobIntent(SwapService.ACTION_PREVIEW))
        }.also(::add)
        previewBefore = ImageView(this).apply { adjustViewBounds = true }
        previewAfter = ImageView(this).apply { adjustViewBounds = true }
        previewRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            fun half(label: String, image: ImageView) = LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(2), 0, dp(2), 0)
                addView(text(label, 12f))
                addView(image, MATCH_PARENT, WRAP_CONTENT)
            }
            addView(half("Before", previewBefore), LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(half("After", previewAfter), LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            visibility = View.GONE
        }.also(::add)

        section("7. Swap the video")

        swapButton = button("Swap faces") { startForegroundService(jobIntent(SwapService.ACTION_SWAP)) }.also(::add)
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 1000
            visibility = View.GONE
        }.also(::add)
        statusText = text("", 13f).also(::add)
        cancelButton = button("Cancel") {
            startService(Intent(this, SwapService::class.java).setAction(SwapService.ACTION_CANCEL))
        }.also(::add)
        resultRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.START
            addView(button("Play") { openResult(Intent.ACTION_VIEW) })
            addView(button("Share") { openResult(Intent.ACTION_SEND) })
            visibility = View.GONE
        }.also(::add)

        val scroll = ScrollView(this).apply {
            addView(column, MATCH_PARENT, WRAP_CONTENT)
        }
        // Android 15+ draws edge to edge for apps targeting it; keep content clear of system bars.
        if (Build.VERSION.SDK_INT >= 35) {
            scroll.setOnApplyWindowInsetsListener { v, insets ->
                val bars = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout())
                v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
                insets
            }
        }
        return scroll
    }

    private fun updateThresholdLabel() {
        thresholdLabel.text = "Match threshold: %.2f".format(thresholdValue())
    }

    companion object {
        private const val PICK_SOURCE = 1
        private const val PICK_VIDEO = 2
        private const val PICK_REFERENCE = 3
        private const val ID_ALL = 101
        private const val ID_LARGEST = 102
        private const val ID_REFERENCE = 103
        private const val KEY_SOURCE = "source"
        private const val KEY_VIDEO = "video"
        private const val KEY_REFERENCE = "reference"
        private const val PREVIEW_US = 5_000_000L
    }
}
