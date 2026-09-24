package io.github.slayer8366.faceswap.core

import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class YuvTest {
    private fun smoothImage(w: Int, h: Int) = RgbImage(w, h).also { img ->
        for (y in 0 until h) for (x in 0 until w) {
            val i = (y * w + x) * 3
            img.data[i] = (x * 255 / w).toByte()
            img.data[i + 1] = (y * 255 / h).toByte()
            img.data[i + 2] = ((x + y) * 127 / (w + h) + 64).toByte()
        }
    }

    /** Planar I420 with row padding, like many decoders. */
    private fun i420(w: Int, h: Int, pad: Int = 8): Triple<Plane, Plane, Plane> =
        Triple(
            Plane(ByteBuffer.allocate((w + pad) * h), w + pad, 1),
            Plane(ByteBuffer.allocate((w / 2 + pad) * h / 2), w / 2 + pad, 1),
            Plane(ByteBuffer.allocate((w / 2 + pad) * h / 2), w / 2 + pad, 1),
        )

    /** Semi-planar NV12: U and V share one interleaved buffer (pixelStride 2). */
    private fun nv12(w: Int, h: Int): Triple<Plane, Plane, Plane> {
        val uv = ByteBuffer.allocate(w * h / 2)
        val v = uv.duplicate().also { it.position(1) }.slice()
        return Triple(Plane(ByteBuffer.allocate(w * h), w, 1), Plane(uv, w, 2), Plane(v, w, 2))
    }

    @Test
    fun knownValuesBt601Limited() {
        val white = RgbImage(2, 2, ByteArray(12) { -1 })
        val (y, u, v) = i420(2, 2)
        rgbToYuv420(white, y, u, v, YuvMatrix.BT601)
        assertEquals(235, y.get(0, 0)); assertEquals(128, u.get(0, 0)); assertEquals(128, v.get(0, 0))
        val black = RgbImage(2, 2)
        rgbToYuv420(black, y, u, v, YuvMatrix.BT601)
        assertEquals(16, y.get(1, 1))
    }

    @Test
    fun roundTripIsCloseForAllLayoutsAndMatrices() {
        val img = smoothImage(64, 48)
        for (layout in listOf(i420(64, 48), nv12(64, 48))) {
            for (m in listOf(YuvMatrix.BT601, YuvMatrix.BT709, YuvMatrix.BT2020, YuvMatrix(0.299, 0.114, true))) {
                val (y, u, v) = layout
                rgbToYuv420(img, y, u, v, m)
                val back = RgbImage(64, 48)
                yuv420ToRgb(y, u, v, m, back)
                var err = 0.0
                for (i in img.data.indices) err += abs((img.data[i].toInt() and 0xFF) - (back.data[i].toInt() and 0xFF))
                val mean = err / img.data.size
                assertTrue(mean < 2.0, "mean abs error $mean")
            }
        }
    }

    @Test
    fun cropOffsetReadsTheRightPixels() {
        val img = smoothImage(16, 16)
        val (y, u, v) = i420(16, 16)
        rgbToYuv420(img, y, u, v, YuvMatrix.BT601)
        val full = RgbImage(16, 16); yuv420ToRgb(y, u, v, YuvMatrix.BT601, full)
        val part = RgbImage(8, 6); yuv420ToRgb(y, u, v, YuvMatrix.BT601, part, left = 4, top = 2)
        for (c in 0 until 3) assertEquals(full[4, 2, c], part[0, 0, c])
    }
}
