package io.github.slayer8366.faceswap

import android.graphics.Bitmap
import android.graphics.Color
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.github.slayer8366.faceswap.core.RgbImage
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

/**
 * Runs the real MediaCodec pipeline on a device or emulator. Frames are
 * checked with MediaMetadataRetriever, an independent decode path, so a
 * colour or rotation bug in the app can't cancel itself out.
 */
@RunWith(AndroidJUnit4::class)
class VideoTranscoderTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val dir = File(context.cacheDir, "transcoder-test").apply { mkdirs() }

    @After
    fun cleanUp() {
        dir.deleteRecursively()
    }

    private fun fill(img: RgbImage, x0: Int, y0: Int, x1: Int, y1: Int, r: Int, g: Int, b: Int) {
        for (y in y0 until y1) for (x in x0 until x1) {
            val i = (y * img.width + x) * 3
            img.data[i] = r.toByte(); img.data[i + 1] = g.toByte(); img.data[i + 2] = b.toByte()
        }
    }

    /** Red left half, blue right half, white square in the coded top-left corner. */
    private fun halves(img: RgbImage) {
        fill(img, 0, 0, img.width / 2, img.height, 200, 40, 40)
        fill(img, img.width / 2, 0, img.width, img.height, 40, 40, 200)
        fill(img, 0, 0, 48, 48, 255, 255, 255)
    }

    private fun frameAt(file: File, timeUs: Long): Bitmap = MediaMetadataRetriever().run {
        try {
            setDataSource(file.path)
            getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)!!
        } finally {
            release()
        }
    }

    private fun assertColor(bmp: Bitmap, x: Int, y: Int, r: Int, g: Int, b: Int, tol: Int = 30) {
        val p = bmp.getPixel(x, y)
        val msg = "pixel ($x,$y) was (${Color.red(p)},${Color.green(p)},${Color.blue(p)}), expected ($r,$g,$b)"
        assertTrue(msg, abs(Color.red(p) - r) <= tol && abs(Color.green(p) - g) <= tol && abs(Color.blue(p) - b) <= tol)
    }

    private fun tracks(file: File): List<MediaFormat> {
        val ex = MediaExtractor()
        try {
            ex.setDataSource(file.path)
            return (0 until ex.trackCount).map { ex.getTrackFormat(it) }
        } finally {
            ex.release()
        }
    }

    private fun run(input: File, processor: FrameProcessor, maxUs: Long? = null): Pair<File, SwapStats> {
        val out = File(dir, "out.mp4")
        val stats = VideoTranscoder(context).run(
            Uri.fromFile(input), out, processor, label = null, maxDurationUs = maxUs,
            progress = { _, _ -> }, cancelled = { false },
        )
        return out to stats
    }

    @Test
    fun transcodeKeepsFramesColoursAndAudio() {
        val input = File(dir, "in.mp4")
        ClipMaker.make(input, 320, 240, frames = 30, fps = 15) { _, img -> halves(img) }

        // Stand-in for the face swap: paint a green square in the middle.
        val (out, stats) = run(input, FrameProcessor { f -> fill(f, 128, 88, 192, 152, 40, 200, 40); 1 })

        assertEquals(30, stats.frames)
        assertEquals(30, stats.faces)
        assertEquals(AudioResult.COPIED, stats.audio)

        val formats = tracks(out)
        val video = formats.single { it.getString(MediaFormat.KEY_MIME)!!.startsWith("video/") }
        assertEquals(320, video.getInteger(MediaFormat.KEY_WIDTH))
        assertEquals(240, video.getInteger(MediaFormat.KEY_HEIGHT))
        val audio = formats.single { it.getString(MediaFormat.KEY_MIME)!!.startsWith("audio/") }
        assertEquals(MediaFormat.MIMETYPE_AUDIO_AAC, audio.getString(MediaFormat.KEY_MIME))
        // Audio is trimmed to the last video frame (29/15 s), so allow a frame and an AAC packet of slack.
        assertEquals(1_933_333.0, audio.getLong(MediaFormat.KEY_DURATION).toDouble(), 120_000.0)

        val bmp = frameAt(out, 1_000_000)
        assertColor(bmp, 60, 200, 200, 40, 40)   // red half
        assertColor(bmp, 260, 200, 40, 40, 200)  // blue half
        assertColor(bmp, 160, 120, 40, 200, 40)  // processor's square
        assertColor(bmp, 20, 20, 255, 255, 255)  // marker
    }

    @Test
    fun rotatedClipComesOutUpright() {
        // Phones store portrait video as landscape frames plus a 90° flag.
        val input = File(dir, "rotated.mp4")
        ClipMaker.make(input, 320, 240, frames = 10, fps = 10, rotation = 90, audio = false) { _, img -> halves(img) }

        // The processor must see upright frames: 240 wide, 320 tall.
        var seen = 0 to 0
        val (out, stats) = run(input, FrameProcessor { f -> seen = f.width to f.height; 0 })
        assertEquals(240 to 320, seen)
        assertEquals(AudioResult.NONE, stats.audio)

        val video = tracks(out).single()
        assertEquals(240, video.getInteger(MediaFormat.KEY_WIDTH))
        assertEquals(320, video.getInteger(MediaFormat.KEY_HEIGHT))
        val rotation = if (video.containsKey(MediaFormat.KEY_ROTATION)) video.getInteger(MediaFormat.KEY_ROTATION) else 0
        assertEquals(0, rotation)

        // Rotating clockwise moves the coded top-left marker to the top-right,
        // the red left half to the top, and the blue right half to the bottom.
        val bmp = frameAt(out, 500_000)
        assertColor(bmp, 220, 20, 255, 255, 255)
        assertColor(bmp, 120, 60, 200, 40, 40)
        assertColor(bmp, 120, 260, 40, 40, 200)

        // The preview path must agree with the output.
        val upright = VideoTranscoder(context).frameAt(Uri.fromFile(input), 500_000)
        assertEquals(240, upright.width)
        assertEquals(320, upright.height)
        assertTrue("preview marker", upright[220, 20, 0] > 220 && upright[220, 20, 2] > 220)
    }

    @Test
    fun frameAtSeeksToTheRequestedTime() {
        // Frame i is a flat grey of level 20 + 10*i, at i/10 s.
        val input = File(dir, "grey.mp4")
        ClipMaker.make(input, 160, 120, frames = 20, fps = 10, audio = false) { i, img ->
            val v = 20 + 10 * i
            fill(img, 0, 0, img.width, img.height, v, v, v)
        }
        val t = VideoTranscoder(context)
        for ((timeUs, index) in listOf(0L to 0, 1_000_000L to 10, 1_450_000L to 15)) {
            val level = t.frameAt(Uri.fromFile(input), timeUs)[80, 60, 1]
            assertEquals("frame at $timeUs us", 20 + 10 * index, level, 4)
        }
    }

    @Test
    fun maxDurationStopsEarly() {
        val input = File(dir, "long.mp4")
        ClipMaker.make(input, 160, 120, frames = 30, fps = 15) { _, img -> halves(img) }
        val (_, stats) = run(input, FrameProcessor { 0 }, maxUs = 1_000_000L)
        // Frames at 0, 1/15 s ... 1 s inclusive.
        assertEquals(16, stats.frames)
    }

    private fun assertEquals(message: String, expected: Int, actual: Int, tolerance: Int) =
        assertTrue("$message: expected $expected ± $tolerance, got $actual", abs(expected - actual) <= tolerance)
}
