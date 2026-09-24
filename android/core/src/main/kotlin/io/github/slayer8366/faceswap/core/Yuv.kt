package io.github.slayer8366.faceswap.core

import java.nio.ByteBuffer

/**
 * One plane of a YUV 4:2:0 image as `android.media.Image.Plane` exposes it.
 * Indexing is absolute from 0, matching MediaCodec's plane buffers.
 */
class Plane(val buffer: ByteBuffer, val rowStride: Int, val pixelStride: Int) {
    fun get(x: Int, y: Int): Int = buffer.get(y * rowStride + x * pixelStride).toInt() and 0xFF
    fun put(x: Int, y: Int, v: Int) {
        buffer.put(y * rowStride + x * pixelStride, v.coerceIn(0, 255).toByte())
    }
}

/**
 * YCbCr colour matrix. [fullRange] false means studio swing (Y 16-235,
 * chroma 16-240), which is what nearly all camera and web video uses.
 */
class YuvMatrix(kr: Double, kb: Double, val fullRange: Boolean) {
    private val kg = 1 - kr - kb
    private val yScale = if (fullRange) 255.0 else 219.0
    private val cScale = if (fullRange) 255.0 else 224.0
    private val yOff = if (fullRange) 0.0 else 16.0

    // Forward, from 0-255 RGB.
    private val fy = doubleArrayOf(kr, kg, kb).map { it * yScale / 255 }
    private val fu = doubleArrayOf(-kr / (2 * (1 - kb)), -kg / (2 * (1 - kb)), 0.5).map { it * cScale / 255 }
    private val fv = doubleArrayOf(0.5, -kg / (2 * (1 - kr)), -kb / (2 * (1 - kr))).map { it * cScale / 255 }

    // Inverse, to 0-255 RGB.
    private val iy = 255 / yScale
    private val rv = 2 * (1 - kr) * 255 / cScale
    private val bu = 2 * (1 - kb) * 255 / cScale
    private val gu = -bu * kb / kg
    private val gv = -rv * kr / kg

    fun toRgb(y: Int, u: Int, v: Int, out: ByteArray, at: Int) {
        val l = (y - yOff) * iy
        val cu = u - 128.0; val cv = v - 128.0
        out[at] = clamp(l + rv * cv)
        out[at + 1] = clamp(l + gu * cu + gv * cv)
        out[at + 2] = clamp(l + bu * cu)
    }

    fun luma(r: Int, g: Int, b: Int) = (yOff + fy[0] * r + fy[1] * g + fy[2] * b + 0.5).toInt()
    fun cb(r: Double, g: Double, b: Double) = (128 + fu[0] * r + fu[1] * g + fu[2] * b + 0.5).toInt()
    fun cr(r: Double, g: Double, b: Double) = (128 + fv[0] * r + fv[1] * g + fv[2] * b + 0.5).toInt()

    private fun clamp(v: Double): Byte = (v + 0.5).toInt().coerceIn(0, 255).toByte()

    companion object {
        val BT601 = YuvMatrix(0.299, 0.114, fullRange = false)
        val BT709 = YuvMatrix(0.2126, 0.0722, fullRange = false)
        val BT2020 = YuvMatrix(0.2627, 0.0593, fullRange = false)
    }
}

/** Convert a 4:2:0 image into [out], reading from ([left], [top]) of the planes. */
fun yuv420ToRgb(
    y: Plane, u: Plane, v: Plane, matrix: YuvMatrix, out: RgbImage, left: Int = 0, top: Int = 0,
) {
    val d = out.data
    for (row in 0 until out.height) {
        val sy = top + row
        for (col in 0 until out.width) {
            val sx = left + col
            matrix.toRgb(y.get(sx, sy), u.get(sx / 2, sy / 2), v.get(sx / 2, sy / 2), d, (row * out.width + col) * 3)
        }
    }
}

/** Write [img] into 4:2:0 planes; chroma is the mean of each 2x2 block. Needs even dimensions. */
fun rgbToYuv420(img: RgbImage, y: Plane, u: Plane, v: Plane, matrix: YuvMatrix) {
    require(img.width % 2 == 0 && img.height % 2 == 0) { "Dimensions must be even" }
    for (row in 0 until img.height) for (col in 0 until img.width) {
        y.put(col, row, matrix.luma(img[col, row, 0], img[col, row, 1], img[col, row, 2]))
    }
    for (row in 0 until img.height / 2) for (col in 0 until img.width / 2) {
        var r = 0.0; var g = 0.0; var b = 0.0
        for (dy in 0..1) for (dx in 0..1) {
            r += img[2 * col + dx, 2 * row + dy, 0]
            g += img[2 * col + dx, 2 * row + dy, 1]
            b += img[2 * col + dx, 2 * row + dy, 2]
        }
        u.put(col, row, matrix.cb(r / 4, g / 4, b / 4))
        v.put(col, row, matrix.cr(r / 4, g / 4, b / 4))
    }
}
