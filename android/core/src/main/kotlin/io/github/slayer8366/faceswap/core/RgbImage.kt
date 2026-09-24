package io.github.slayer8366.faceswap.core

import kotlin.math.floor
import kotlin.math.roundToInt

/** 8-bit interleaved RGB image, row-major with no row padding. */
class RgbImage(
    val width: Int,
    val height: Int,
    val data: ByteArray = ByteArray(width * height * 3),
) {
    init {
        require(width > 0 && height > 0) { "Image must be non-empty, got ${width}x$height" }
        require(data.size == width * height * 3) { "Expected ${width * height * 3} bytes, got ${data.size}" }
    }

    operator fun get(x: Int, y: Int, c: Int): Int = data[(y * width + x) * 3 + c].toInt() and 0xFF

    fun copy() = RgbImage(width, height, data.copyOf())

    /** Rotate clockwise by 0, 90, 180 or 270 degrees. */
    fun rotate(degrees: Int): RgbImage {
        val d = ((degrees % 360) + 360) % 360
        if (d == 0) return this
        require(d % 90 == 0) { "Rotation must be a multiple of 90, got $degrees" }
        val (ow, oh) = if (d == 180) width to height else height to width
        val out = RgbImage(ow, oh)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val (nx, ny) = when (d) {
                    90 -> (height - 1 - y) to x
                    180 -> (width - 1 - x) to (height - 1 - y)
                    else -> y to (width - 1 - x)
                }
                System.arraycopy(data, (y * width + x) * 3, out.data, (ny * ow + nx) * 3, 3)
            }
        }
        return out
    }

    fun crop(x: Int, y: Int, w: Int, h: Int): RgbImage {
        require(x >= 0 && y >= 0 && x + w <= width && y + h <= height) { "Crop out of bounds" }
        val out = RgbImage(w, h)
        for (row in 0 until h) {
            System.arraycopy(data, ((y + row) * width + x) * 3, out.data, row * w * 3, w * 3)
        }
        return out
    }

    /** Add a black border of [pad] pixels on every side. */
    fun pad(pad: Int): RgbImage {
        val out = RgbImage(width + 2 * pad, height + 2 * pad)
        for (row in 0 until height) {
            System.arraycopy(data, row * width * 3, out.data, ((row + pad) * out.width + pad) * 3, width * 3)
        }
        return out
    }

    /** Bilinear resize with OpenCV's INTER_LINEAR pixel-centre convention. */
    fun resize(w: Int, h: Int): RgbImage {
        if (w == width && h == height) return copy()
        val out = RgbImage(w, h)
        val sx = width.toDouble() / w
        val sy = height.toDouble() / h
        val x0 = IntArray(w)
        val fx = FloatArray(w)
        for (x in 0 until w) {
            val (i, f) = sampleCoord((x + 0.5) * sx - 0.5, width)
            x0[x] = i; fx[x] = f
        }
        for (y in 0 until h) {
            val (y0, fy) = sampleCoord((y + 0.5) * sy - 0.5, height)
            val y1 = minOf(y0 + 1, height - 1)
            for (x in 0 until w) {
                val x1 = minOf(x0[x] + 1, width - 1)
                for (c in 0 until 3) {
                    val top = this[x0[x], y0, c] * (1 - fx[x]) + this[x1, y0, c] * fx[x]
                    val bot = this[x0[x], y1, c] * (1 - fx[x]) + this[x1, y1, c] * fx[x]
                    out.data[(y * w + x) * 3 + c] = (top * (1 - fy) + bot * fy).roundToInt().toByte()
                }
            }
        }
        return out
    }

    /**
     * Alpha-blend non-premultiplied ARGB pixels (as returned by
     * `Bitmap.getPixels`) onto this image with their top-left at ([x], [y]).
     */
    fun blendArgb(pixels: IntArray, w: Int, h: Int, x: Int, y: Int) {
        require(pixels.size >= w * h)
        for (row in 0 until h) {
            val ty = y + row
            if (ty !in 0 until height) continue
            for (col in 0 until w) {
                val tx = x + col
                if (tx !in 0 until width) continue
                val p = pixels[row * w + col]
                val a = (p ushr 24) and 0xFF
                if (a == 0) continue
                val i = (ty * width + tx) * 3
                val src = intArrayOf((p shr 16) and 0xFF, (p shr 8) and 0xFF, p and 0xFF)
                for (c in 0 until 3) {
                    val dst = data[i + c].toInt() and 0xFF
                    data[i + c] = ((src[c] * a + dst * (255 - a) + 127) / 255).toByte()
                }
            }
        }
    }

    private fun sampleCoord(f: Double, size: Int): Pair<Int, Float> {
        var i = floor(f).toInt()
        var frac = (f - i).toFloat()
        if (i < 0) { i = 0; frac = 0f }
        if (i >= size - 1) { i = size - 1; frac = 0f }
        return i to frac
    }
}
