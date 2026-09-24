package io.github.slayer8366.faceswap.core

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.DataInputStream
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * Checks the published fast-mode files from tools/quantize_swapper.py:
 * the emap file equals the fp32 model's, and the int8 swapper stays close
 * to the fp32 reference output. Needs FACESWAP_MODEL_DIR, FACESWAP_GOLDEN_DIR
 * and FACESWAP_INT8_DIR (holding inswapper_128_int8.onnx and inswapper_emap.bin).
 */
class Int8SwapperTest {
    private val modelDir = System.getenv("FACESWAP_MODEL_DIR")?.let { File(it, "models") }
    private val goldenDir = System.getenv("FACESWAP_GOLDEN_DIR")?.let(::File)
    private val int8Dir = System.getenv("FACESWAP_INT8_DIR")?.let(::File)

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
    fun int8SwapperMatchesFp32Closely() {
        if (modelDir == null || goldenDir == null || int8Dir?.resolve("inswapper_128_int8.onnx")?.exists() != true) {
            println("SKIPPED: set FACESWAP_MODEL_DIR, FACESWAP_GOLDEN_DIR and FACESWAP_INT8_DIR")
            return
        }
        val fp32Emap = OnnxInitializers.lastInitializer(modelDir.resolve("inswapper_128.onnx")).data
        // Same reader as ModelStore's emap cache: big-endian floats.
        val emap = DataInputStream(int8Dir.resolve("inswapper_emap.bin").inputStream().buffered()).use { s ->
            FloatArray(512 * 512) { s.readFloat() }
        }
        assertContentEquals(fp32Emap, emap, "emap file differs from the fp32 model's")

        val env = OrtEnvironment.getEnvironment()
        OrtSession.SessionOptions().use { opts ->
            env.createSession(int8Dir.resolve("inswapper_128_int8.onnx").absolutePath, opts).use { session ->
                val swapper = InSwapper(env, session, emap)
                val frame = load(goldenDir.resolve("frame.png"))
                val golden = Json.parse(goldenDir.resolve("golden.json").readText()) as Map<*, *>
                @Suppress("UNCHECKED_CAST")
                fun floats(v: Any?) = (v as List<Number>).map { it.toFloat() }.toFloatArray()
                val kps = floats(((golden["faces"] as List<*>)[0] as Map<*, *>)["kps"])
                val latent = swapper.latent(floats(golden["source_embedding"]))
                val (fake, _) = swapper.generate(frame, kps, latent)
                val ref = load(goldenDir.resolve("fake_one.png"))
                var s = 0L
                for (i in fake.data.indices) s += abs((fake.data[i].toInt() and 0xFF) - (ref.data[i].toInt() and 0xFF))
                val diff = s.toDouble() / fake.data.size
                val t0 = System.nanoTime(); repeat(3) { swapper.generate(frame, kps, latent) }
                println("int8 vs fp32 generated face: mean abs diff %.2f / 255; %.0f ms per face (Kotlin/ORT Java)"
                    .format(diff, (System.nanoTime() - t0) / 3e6))
                assertTrue(diff < 4.0, "int8 output differs from fp32 by $diff")
            }
        }
    }
}
