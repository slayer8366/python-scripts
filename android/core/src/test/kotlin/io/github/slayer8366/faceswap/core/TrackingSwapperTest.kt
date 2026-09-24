package io.github.slayer8366.faceswap.core

import java.io.File
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tracking vs full detection on every frame, with real models, on a clip
 * panning across the golden frame. Needs FACESWAP_MODEL_DIR and FACESWAP_GOLDEN_DIR.
 */
class TrackingSwapperTest {
    private val modelDir = System.getenv("FACESWAP_MODEL_DIR")?.let { File(it, "models") }
    private val goldenDir = System.getenv("FACESWAP_GOLDEN_DIR")?.let(::File)
    private val enabled = modelDir?.resolve("inswapper_128.onnx")?.exists() == true && goldenDir?.isDirectory == true
    private var engine: FaceEngine? = null

    @AfterTest
    fun tearDown() { engine?.close() }

    private fun load(file: File): RgbImage {
        val bi = ImageIO.read(file)
        val img = RgbImage(bi.width, bi.height)
        for (y in 0 until bi.height) for (x in 0 until bi.width) {
            val p = bi.getRGB(x, y); val i = (y * bi.width + x) * 3
            img.data[i] = (p shr 16).toByte(); img.data[i + 1] = (p shr 8).toByte(); img.data[i + 2] = p.toByte()
        }
        return img
    }

    @Test
    fun trackingMatchesFullDetection() {
        if (!enabled) { println("SKIPPED: set FACESWAP_MODEL_DIR and FACESWAP_GOLDEN_DIR"); return }
        val swap = modelDir!!.resolve("inswapper_128.onnx")
        val e = FaceEngine(ModelFiles(modelDir.resolve("buffalo_l/det_10g.onnx"), modelDir.resolve("buffalo_l/w600k_r50.onnx"),
            swap, OnnxInitializers.lastInitializer(swap).data)).also { engine = it }
        val base = load(goldenDir!!.resolve("frame.png")).pad(40)
        val frames = (0 until 12).map { i -> base.crop(3 * i, 2 * i, 640, 480) }
        val latent = e.swapper.latent(e.faceFromStill(load(goldenDir.resolve("source.png"))).embedding!!)

        // 1. Landmarks: tracked vs a fresh full detection on the same frame.
        val tracker = TrackingSwapper(e, latent, SwapMode.ALL, null, 0.35f, cropTracking = true, fullEvery = 6).tracker
        var worst = 0f
        frames.forEachIndexed { i, f ->
            val tracked = tracker.update(f).map { it.face }
            val full = e.detect(f)
            assertEquals(full.size, tracked.size, "face count at frame $i")
            for (t in tracked) {
                val m = full.maxBy { iou(it.bbox, t.bbox) }
                for (k in 0 until 10) worst = maxOf(worst, abs(m.kps[k] - t.kps[k]))
            }
        }
        // Baseline: the full detector's own jitter when the frame moves by 1 px.
        var jitter = 0f
        frames.indices.forEach { i ->
            val a = e.detect(frames[i]); val b = e.detect(base.crop(3 * i + 1, 2 * i + 1, 640, 480))
            for (fa in a) {
                val fb = b.maxBy { iou(it.bbox, fa.bbox) }
                for (k in 0 until 10) jitter = maxOf(jitter, abs(fa.kps[k] - (fb.kps[k] + 1f)))
            }
        }
        println("tracked vs full-detection landmarks: worst error %.2f px over %d frames; %d full, %d crop detections"
            .format(worst, frames.size, tracker.fullDetections, tracker.cropDetections))
        println("full detector's own jitter under a 1 px shift: worst %.2f px".format(jitter))
        assertEquals(2, tracker.fullDetections)
        assertTrue(worst < 3.0f, "landmark error $worst px")

        // 2. Output and time: tracking swapper vs swapping with full detection every frame.
        val ts = TrackingSwapper(e, latent, SwapMode.ALL, null, 0.35f, cropTracking = true, fullEvery = 6)
        var tTrack = 0L; var tFull = 0L; var worstDiff = 0.0
        for (f in frames) {
            val a = f.copy(); val b = f.copy()
            var t0 = System.nanoTime(); ts.process(a); tTrack += System.nanoTime() - t0
            t0 = System.nanoTime(); e.swapFrame(b, latent, SwapMode.ALL); tFull += System.nanoTime() - t0
            var s = 0L
            for (k in a.data.indices) s += abs((a.data[k].toInt() and 0xFF) - (b.data[k].toInt() and 0xFF))
            worstDiff = maxOf(worstDiff, s.toDouble() / a.data.size)
        }
        println("swapped frames, tracking vs full: worst mean abs diff %.3f / 255; %.0f vs %.0f ms per frame"
            .format(worstDiff, tTrack / 1e6 / frames.size, tFull / 1e6 / frames.size))
        assertTrue(worstDiff < 0.5, "tracked swap differs by $worstDiff")

        // 3. Reference mode embeds far less often, with or without crop tracking.
        val ref = e.recognizer.embed(frames[0], e.detect(frames[0])[1].kps)
        for (crop in listOf(false, true)) {
            val rs = TrackingSwapper(e, latent, SwapMode.REFERENCE, ref, 0.35f, cropTracking = crop)
            val swapped = frames.map { rs.process(it.copy()) }
            println("reference mode, cropTracking=$crop: $swapped faces swapped, ${rs.embeddings} embeddings " +
                "(no caching: ${3 * frames.size}); ${rs.tracker.fullDetections} full detections")
            assertTrue(swapped.all { it == 1 })
            assertEquals(6, rs.embeddings) // 3 faces, checked at frames 0 and 6
            assertEquals(if (crop) 2 else frames.size, rs.tracker.fullDetections)
        }
    }

    /** The common case: one face in a high-resolution frame, with the int8 swapper. Needs FACESWAP_INT8_DIR too. */
    @Test
    fun oneFaceTimingWithInt8() {
        val int8 = System.getenv("FACESWAP_INT8_DIR")?.let(::File)?.resolve("inswapper_128_int8.onnx")
        if (!enabled || int8?.exists() != true) { println("SKIPPED: also needs FACESWAP_INT8_DIR"); return }
        val fp32 = modelDir!!.resolve("inswapper_128.onnx")
        val e = FaceEngine(ModelFiles(modelDir.resolve("buffalo_l/det_10g.onnx"), modelDir.resolve("buffalo_l/w600k_r50.onnx"),
            int8, OnnxInitializers.lastInitializer(fp32).data)).also { engine = it }
        val frame = load(goldenDir!!.resolve("frame.png"))
        val f0 = e.detect(frame).first()
        // A 240x320 window around one face, scaled 3x to 720x960, panned 6 px per frame.
        val cx = ((f0.bbox[0] + f0.bbox[2]) / 2).toInt(); val cy = ((f0.bbox[1] + f0.bbox[3]) / 2).toInt()
        val big = frame.crop((cx - 130).coerceAtLeast(0), (cy - 170).coerceAtLeast(0), 260, 340).resize(780, 1020)
        val frames = (0 until 12).map { i -> big.crop(3 * i, 2 * i, 720, 960) }
        val latent = e.swapper.latent(e.faceFromStill(load(goldenDir.resolve("source.png"))).embedding!!)
        val ts = TrackingSwapper(e, latent, SwapMode.ALL, null, 0.35f, cropTracking = true, fullEvery = 6)
        // Warm up both paths.
        ts.process(frames[0].copy()); e.swapFrame(frames[0].copy(), latent, SwapMode.ALL)
        val ts2 = TrackingSwapper(e, latent, SwapMode.ALL, null, 0.35f, cropTracking = true, fullEvery = 6)
        var tTrack = 0L; var tFull = 0L; var n = 0
        for (f in frames) {
            var t0 = System.nanoTime(); n += ts2.process(f.copy()); tTrack += System.nanoTime() - t0
            t0 = System.nanoTime(); e.swapFrame(f.copy(), latent, SwapMode.ALL); tFull += System.nanoTime() - t0
        }
        println("one face, 720x960, int8: tracking %.0f ms vs full detection %.0f ms per frame (%d faces swapped in %d frames)"
            .format(tTrack / 1e6 / frames.size, tFull / 1e6 / frames.size, n, frames.size))
        assertEquals(frames.size, n)
    }
}
