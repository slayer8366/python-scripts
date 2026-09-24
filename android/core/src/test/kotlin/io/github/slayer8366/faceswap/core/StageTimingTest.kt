package io.github.slayer8366.faceswap.core

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import javax.imageio.ImageIO
import kotlin.test.Test

/**
 * Where does the time per frame go? Prints median milliseconds per stage.
 * Runs only with FACESWAP_BENCH=1 plus the model and golden directories.
 * Desktop CPU numbers; phones will differ in scale, not necessarily in ratio.
 */
class StageTimingTest {
    private val modelDir = System.getenv("FACESWAP_MODEL_DIR")?.let { File(it, "models") }
    private val goldenDir = System.getenv("FACESWAP_GOLDEN_DIR")?.let(::File)

    private fun load(file: File): RgbImage {
        val bi = ImageIO.read(file)
        val img = RgbImage(bi.width, bi.height)
        for (y in 0 until bi.height) for (x in 0 until bi.width) {
            val p = bi.getRGB(x, y); val i = (y * bi.width + x) * 3
            img.data[i] = (p shr 16).toByte(); img.data[i + 1] = (p shr 8).toByte(); img.data[i + 2] = p.toByte()
        }
        return img
    }

    private fun median(reps: Int = 7, block: () -> Unit): Double {
        block() // warm-up
        val times = (0 until reps).map { val t = System.nanoTime(); block(); (System.nanoTime() - t) / 1e6 }
        return times.sorted()[reps / 2]
    }

    @Test
    fun printStageTimings() {
        if (System.getenv("FACESWAP_BENCH") != "1" || modelDir == null || goldenDir == null) {
            println("SKIPPED: set FACESWAP_BENCH=1, FACESWAP_MODEL_DIR and FACESWAP_GOLDEN_DIR")
            return
        }
        val swap = modelDir.resolve("inswapper_128.onnx")
        FaceEngine(
            ModelFiles(
                modelDir.resolve("buffalo_l/det_10g.onnx"),
                modelDir.resolve("buffalo_l/w600k_r50.onnx"),
                swap,
                OnnxInitializers.lastInitializer(swap).data,
            ),
        ).use { engine ->
            val frame = load(goldenDir.resolve("frame.png"))
            val faces = engine.detect(frame)
            val face = faces.first()
            val latent = engine.swapper.latent(engine.faceFromStill(load(goldenDir.resolve("source.png"))).embedding!!)

            val detect = median { engine.detect(frame) }
            val embed = median { engine.recognizer.embed(frame, face.kps) }
            val (fake, m) = engine.swapper.generate(frame, face.kps, latent)
            val generate = median { engine.swapper.generate(frame, face.kps, latent) }
            val paste = median { engine.swapper.pasteBack(frame.copy(), fake, m) }

            // Candidate: re-detect in a small crop around the known face.
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions()
            val session = env.createSession(modelDir.resolve("buffalo_l/det_10g.onnx").absolutePath, opts)
            val smallDetectors = listOf(128, 160, 192).map { it to ScrfdDetector(env, session, inputSize = it) }
            val w = face.bbox[2] - face.bbox[0]; val h = face.bbox[3] - face.bbox[1]
            val side = (maxOf(w, h) * 2f).toInt()
            val cx = ((face.bbox[0] + face.bbox[2]) / 2).toInt(); val cy = ((face.bbox[1] + face.bbox[3]) / 2).toInt()
            val x0 = (cx - side / 2).coerceIn(0, frame.width - 1); val y0 = (cy - side / 2).coerceIn(0, frame.height - 1)
            val crop = frame.crop(x0, y0, minOf(side, frame.width - x0), minOf(side, frame.height - y0))

            println("frame ${frame.width}x${frame.height}, ${faces.size} faces, face box ${w.toInt()}x${h.toInt()}, crop ${crop.width}x${crop.height}")
            println("detect @640 full frame : %7.1f ms".format(detect))
            for ((size, det) in smallDetectors) {
                println("detect @%-3d on crop     : %7.1f ms (found %d)".format(size, median { det.detect(crop) }, det.detect(crop).size))
            }
            println("embed (ArcFace) / face : %7.1f ms".format(embed))
            println("swap generate / face   : %7.1f ms".format(generate))
            println("paste-back / face      : %7.1f ms".format(paste))
            session.close(); opts.close()
        }
    }
}
