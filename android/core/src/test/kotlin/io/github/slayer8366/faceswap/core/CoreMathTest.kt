package io.github.slayer8366.faceswap.core

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class CoreMathTest {
    private fun randomImage(w: Int, h: Int, seed: Int = 1) =
        RgbImage(w, h, Random(seed).nextBytes(w * h * 3))

    @Test
    fun similarityRecoversKnownTransform() {
        val s = 1.7; val t = 0.4
        val truth = Affine(s * cos(t), -s * sin(t), 12.0, s * sin(t), s * cos(t), -5.0)
        val src = floatArrayOf(10f, 20f, 40f, 22f, 25f, 35f, 12f, 50f, 38f, 52f)
        val dst = DoubleArray(10) { i ->
            val x = src[i - i % 2].toDouble(); val y = src[i - i % 2 + 1].toDouble()
            if (i % 2 == 0) truth.mapX(x, y) else truth.mapY(x, y)
        }
        val m = similarityTransform(src, dst)
        truth.toArray().zip(m.toArray()).forEach { (a, b) -> assertEquals(a, b, 1e-4) }
    }

    @Test
    fun affineInverseRoundTrips() {
        val m = Affine(1.2, 0.3, 5.0, -0.2, 0.9, 7.0)
        val inv = m.invert()
        val x = inv.mapX(m.mapX(3.0, 4.0), m.mapY(3.0, 4.0))
        val y = inv.mapY(m.mapX(3.0, 4.0), m.mapY(3.0, 4.0))
        assertEquals(3.0, x, 1e-9); assertEquals(4.0, y, 1e-9)
    }

    @Test
    fun rotateMovesPixelsClockwise() {
        val img = RgbImage(3, 2)
        img.data[0] = 9 // (0,0) red
        val r = img.rotate(90)
        assertEquals(2, r.width); assertEquals(3, r.height)
        assertEquals(9, r[1, 0, 0]) // top-left goes to top-right
        val src = randomImage(5, 3)
        assertContentEquals(src.data, src.rotate(90).rotate(90).rotate(90).rotate(90).data)
        assertContentEquals(src.rotate(180).data, src.rotate(90).rotate(90).data)
        assertContentEquals(src.rotate(270).data, src.rotate(-90).data)
    }

    @Test
    fun cropAndPad() {
        val src = randomImage(6, 4)
        val padded = src.pad(2)
        assertEquals(10, padded.width)
        assertEquals(0, padded[0, 0, 1])
        assertContentEquals(src.data, padded.crop(2, 2, 6, 4).data)
    }

    @Test
    fun resizeIdentityAndConstant() {
        val src = randomImage(7, 5)
        assertContentEquals(src.data, src.resize(7, 5).data)
        val grey = RgbImage(40, 30, ByteArray(40 * 30 * 3) { 77 })
        assertTrue(grey.resize(13, 9).data.all { it.toInt() == 77 })
    }

    @Test
    fun warpWithIdentityCopiesImage() {
        val src = randomImage(9, 6)
        assertContentEquals(src.data, warpAffine(src, Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), 9, 6).data)
        val shifted = warpAffine(src, Affine(1.0, 0.0, 2.0, 0.0, 1.0, 0.0), 9, 6)
        assertEquals(0, shifted[0, 0, 0])
        assertEquals(src[0, 3, 1], shifted[2, 3, 1])
    }

    @Test
    fun blendArgbAlpha() {
        val img = RgbImage(2, 1, byteArrayOf(100, 100, 100, 100, 100, 100))
        img.blendArgb(intArrayOf(0x80FFFFFF.toInt(), 0x00000000), 2, 1, 0, 0)
        assertEquals(178, img[0, 0, 0]) // (255*128 + 100*127 + 127) / 255
        assertEquals(100, img[1, 0, 0])
    }

    @Test
    fun erodeMatchesBruteForce() {
        val w = 23; val h = 17
        val rnd = Random(3)
        val src = FloatArray(w * h) { rnd.nextFloat() }
        for (k in listOf(1, 4, 5, 10)) {
            val out = erode(src, w, h, k)
            for (y in 0 until h) for (x in 0 until w) {
                var m = Float.MAX_VALUE
                for (dy in -(k / 2)..(k - 1 - k / 2)) for (dx in -(k / 2)..(k - 1 - k / 2)) {
                    val xx = x + dx; val yy = y + dy
                    if (xx in 0 until w && yy in 0 until h) m = minOf(m, src[yy * w + xx])
                }
                assertEquals(m, out[y * w + x], "k=$k at $x,$y")
            }
        }
    }

    @Test
    fun gaussianKernelMatchesOpenCvSigmaRule() {
        val k = gaussianKernel(11) // OpenCV: sigma = 0.3*((11-1)*0.5-1)+0.8 = 2.0
        assertEquals(1.0, k.sum().toDouble(), 1e-6)
        assertEquals(exp(-25.0 / 8.0) / exp(0.0), (k[0] / k[5]).toDouble(), 1e-5)
        val flat = FloatArray(20 * 20) { 3f }
        assertTrue(gaussianBlur(flat, 20, 20, 11).all { abs(it - 3f) < 1e-4 })
    }

    private fun face(x1: Float, y1: Float, x2: Float, y2: Float, emb: FloatArray? = null, score: Float = 0.9f) =
        Face(floatArrayOf(x1, y1, x2, y2), score, FloatArray(10)).also { it.embedding = emb }

    @Test
    fun selectTargetsModes() {
        val small = face(0f, 0f, 10f, 10f); val big = face(0f, 0f, 50f, 40f)
        assertEquals(listOf(small, big), selectTargets(listOf(small, big), SwapMode.ALL))
        assertEquals(listOf(big), selectTargets(listOf(small, big), SwapMode.LARGEST))
        assertEquals(emptyList(), selectTargets(emptyList(), SwapMode.LARGEST))
        val same = face(0f, 0f, 1f, 1f, floatArrayOf(0.9f, 0.1f))
        val other = face(0f, 0f, 1f, 1f, floatArrayOf(0f, 1f))
        val none = face(0f, 0f, 1f, 1f)
        assertEquals(listOf(same), selectTargets(listOf(same, other, none), SwapMode.REFERENCE, floatArrayOf(1f, 0f), 0.35f))
        assertFailsWith<IllegalArgumentException> { selectTargets(listOf(same), SwapMode.REFERENCE) }
    }

    @Test
    fun nmsSuppressesOverlaps() {
        val a = face(0f, 0f, 100f, 100f, score = 0.9f)
        val b = face(5f, 5f, 105f, 105f, score = 0.8f)
        val c = face(200f, 200f, 250f, 250f, score = 0.7f)
        assertEquals(listOf(a, c), nms(listOf(a, b, c), 0.4f))
    }
}
