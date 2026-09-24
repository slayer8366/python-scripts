package io.github.slayer8366.faceswap.core

import java.io.File
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Compares the Kotlin port with the Python/insightface pipeline on the same
 * inputs. Needs FACESWAP_MODEL_DIR (with models/inswapper_128.onnx and
 * models/buffalo_l/) and FACESWAP_GOLDEN_DIR (from tools/make_golden.py);
 * each test returns early without them.
 */
class GoldenTest {
    private val modelDir = System.getenv("FACESWAP_MODEL_DIR")?.let { File(it, "models") }
    private val goldenDir = System.getenv("FACESWAP_GOLDEN_DIR")?.let(::File)
    private val enabled = modelDir?.resolve("inswapper_128.onnx")?.exists() == true && goldenDir?.isDirectory == true

    private val engine by lazy {
        val swap = modelDir!!.resolve("inswapper_128.onnx")
        FaceEngine(
            ModelFiles(
                modelDir.resolve("buffalo_l/det_10g.onnx"),
                modelDir.resolve("buffalo_l/w600k_r50.onnx"),
                swap,
                OnnxInitializers.lastInitializer(swap).data,
            ),
        )
    }
    private var engineUsed = false

    @AfterTest
    fun tearDown() { if (engineUsed) engine.close() }

    private fun skip(): Boolean {
        if (!enabled) println("SKIPPED: set FACESWAP_MODEL_DIR and FACESWAP_GOLDEN_DIR")
        else engineUsed = true
        return !enabled
    }

    private fun load(name: String): RgbImage {
        val bi = ImageIO.read(goldenDir!!.resolve(name))
        val img = RgbImage(bi.width, bi.height)
        for (y in 0 until bi.height) for (x in 0 until bi.width) {
            val p = bi.getRGB(x, y); val i = (y * bi.width + x) * 3
            img.data[i] = (p shr 16).toByte(); img.data[i + 1] = (p shr 8).toByte(); img.data[i + 2] = p.toByte()
        }
        return img
    }

    private val golden by lazy { Json.parse(goldenDir!!.resolve("golden.json").readText()) as Map<*, *> }

    @Suppress("UNCHECKED_CAST")
    private fun floats(v: Any?) = (v as List<Number>).map { it.toFloat() }.toFloatArray()

    private fun pyFaces() = (golden["faces"] as List<*>).map { it as Map<*, *> }

    private fun meanAbsDiff(a: RgbImage, b: RgbImage): Double {
        assertEquals(a.width, b.width); assertEquals(a.height, b.height)
        var s = 0L
        for (i in a.data.indices) s += abs((a.data[i].toInt() and 0xFF) - (b.data[i].toInt() and 0xFF))
        return s.toDouble() / a.data.size
    }

    @Test
    fun emapMatchesNumpy() {
        if (skip()) return
        val t = OnnxInitializers.lastInitializer(modelDir!!.resolve("inswapper_128.onnx"))
        assertEquals(listOf(512L, 512L), t.dims.toList())
        floats(golden["emap_first"]).forEachIndexed { i, v -> assertEquals(v, t.data[i], 0f) }
        assertEquals((golden["emap_sum"] as Number).toDouble(), t.data.sumOf { it.toDouble() }, 1e-6)
    }

    @Test
    fun detectionMatchesPython() {
        if (skip()) return
        val faces = engine.detect(load("frame.png"))
        val py = pyFaces()
        assertEquals(py.size, faces.size, "face count")
        faces.zip(py).forEachIndexed { i, (k, p) ->
            val bbox = floats(p["bbox"]); val kps = floats(p["kps"])
            for (j in 0 until 4) assertEquals(bbox[j], k.bbox[j], 1.5f, "face $i bbox[$j]")
            for (j in 0 until 10) assertEquals(kps[j], k.kps[j], 1.5f, "face $i kps[$j]")
            assertEquals((p["score"] as Number).toFloat(), k.score, 0.02f)
        }
    }

    @Test
    fun alignmentMatchesSkimage() {
        if (skip()) return
        val kps = floats(pyFaces()[0]["kps"])
        for ((size, key) in listOf(112 to "M112", 128 to "M128")) {
            val m = estimateNorm(kps, size).toArray()
            floats(golden[key]).forEachIndexed { i, v -> assertEquals(v.toDouble(), m[i], 1e-3, "$key[$i]") }
        }
        val crop = warpAffine(load("frame.png"), estimateNorm(kps, 112), 112, 112)
        val d = meanAbsDiff(crop, load("crop112.png"))
        println("crop112 mean abs diff vs OpenCV: $d")
        assertTrue(d < 1.0, "crop differs by $d")
    }

    @Test
    fun embeddingsMatchPython() {
        if (skip()) return
        val frame = load("frame.png")
        pyFaces().forEachIndexed { i, p ->
            val e = engine.recognizer.embed(frame, floats(p["kps"]))
            val sim = cosineSimilarity(e, floats(p["embedding"]))
            println("face $i embedding cosine vs Python: $sim")
            assertTrue(sim > 0.99f, "face $i cosine $sim")
        }
        val src = engine.faceFromStill(load("source.png"), "source")
        val sim = cosineSimilarity(src.embedding!!, floats(golden["source_embedding"]))
        println("source embedding cosine vs Python (padded retry path): $sim")
        assertTrue(sim > 0.99f)
    }

    @Test
    fun swapMatchesPython() {
        if (skip()) return
        val frame = load("frame.png")
        val latent = engine.swapper.latent(floats(golden["source_embedding"]))
        val kps0 = floats(pyFaces()[0]["kps"])

        val (fake, m) = engine.swapper.generate(frame, kps0, latent)
        val dFake = meanAbsDiff(fake, load("fake_one.png"))
        println("128px swap output mean abs diff vs Python: $dFake")
        assertTrue(dFake < 2.0, "generated face differs by $dFake")

        val one = frame.copy()
        engine.swapper.pasteBack(one, load("fake_one.png"), m)
        val dPaste = meanAbsDiff(one, load("swapped_one.png"))
        println("paste-back (same generated face) mean abs diff vs Python: $dPaste")
        assertTrue(dPaste < 0.1, "paste-back differs by $dPaste")

        val all = frame.copy()
        pyFaces().forEach { engine.swapper.swap(all, floats(it["kps"]), latent) }
        val dAll = meanAbsDiff(all, load("swapped.png"))
        println("full frame, all faces, mean abs diff vs Python: $dAll")
        assertTrue(dAll < 0.5, "full swap differs by $dAll")
    }

    @Test
    fun swapFrameEndToEnd() {
        if (skip()) return
        val frame = load("frame.png")
        val src = engine.faceFromStill(load("source.png"), "source")
        val latent = engine.swapper.latent(src.embedding!!)
        assertEquals(pyFaces().size, engine.swapFrame(frame.copy(), latent, SwapMode.ALL))
        assertEquals(1, engine.swapFrame(frame.copy(), latent, SwapMode.LARGEST))
        val ref = engine.recognizer.embed(frame, floats(pyFaces()[1]["kps"]))
        assertEquals(1, engine.swapFrame(frame.copy(), latent, SwapMode.REFERENCE, ref, 0.35f))
    }
}

/** Just enough JSON for golden.json: objects, arrays, numbers, strings. */
private object Json {
    fun parse(s: String): Any? = Parser(s).value()

    private class Parser(val s: String) {
        var i = 0

        fun value(): Any? {
            skipWs()
            return when (s[i]) {
                '{' -> obj()
                '[' -> array()
                '"' -> string()
                else -> number()
            }
        }

        private fun obj(): Map<String, Any?> {
            val m = LinkedHashMap<String, Any?>()
            i++
            skipWs()
            if (s[i] == '}') { i++; return m }
            while (true) {
                skipWs()
                val key = string()
                skipWs(); expect(':')
                m[key] = value()
                skipWs()
                if (s[i++] == '}') return m
            }
        }

        private fun array(): List<Any?> {
            val l = ArrayList<Any?>()
            i++
            skipWs()
            if (s[i] == ']') { i++; return l }
            while (true) {
                l.add(value())
                skipWs()
                if (s[i++] == ']') return l
            }
        }

        private fun string(): String {
            expect('"')
            val end = s.indexOf('"', i)
            return s.substring(i, end).also { i = end + 1 }
        }

        private fun number(): Double {
            val start = i
            while (i < s.length && s[i] in "+-.eE0123456789") i++
            return s.substring(start, i).toDouble()
        }

        private fun expect(c: Char) { check(s[i] == c) { "Expected $c at $i" }; i++ }

        private fun skipWs() { while (i < s.length && s[i].isWhitespace()) i++ }
    }
}
