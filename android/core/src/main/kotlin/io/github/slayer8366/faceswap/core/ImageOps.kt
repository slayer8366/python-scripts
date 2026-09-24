package io.github.slayer8366.faceswap.core

import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.roundToInt

/**
 * Warp [src] through [m] (source to destination, as with `cv2.warpAffine`)
 * into an [outW] x [outH] image. Bilinear sampling, pixels outside the
 * source read as 0.
 */
fun warpAffine(src: RgbImage, m: Affine, outW: Int, outH: Int): RgbImage {
    val out = RgbImage(outW, outH)
    warpInto(src, m.invert(), Rect(0, 0, outW, outH)) { i, x, y, v -> out.data[(y * outW + x) * 3 + i] = v.roundToInt().toByte() }
    return out
}

/**
 * Bilinearly sample [src] at inv(x, y) for every destination pixel in [roi]
 * and hand each channel value to [sink].
 */
internal inline fun warpInto(src: RgbImage, inv: Affine, roi: Rect, sink: (Int, Int, Int, Float) -> Unit) {
    val w = src.width; val h = src.height; val d = src.data
    for (y in roi.y0 until roi.y1) {
        for (x in roi.x0 until roi.x1) {
            val sx = inv.mapX(x.toDouble(), y.toDouble())
            val sy = inv.mapY(x.toDouble(), y.toDouble())
            val x0 = floor(sx).toInt(); val y0 = floor(sy).toInt()
            val fx = (sx - x0).toFloat(); val fy = (sy - y0).toFloat()
            val w00 = (1 - fx) * (1 - fy); val w10 = fx * (1 - fy)
            val w01 = (1 - fx) * fy; val w11 = fx * fy
            val in00 = x0 in 0 until w && y0 in 0 until h
            val in10 = x0 + 1 in 0 until w && y0 in 0 until h
            val in01 = x0 in 0 until w && y0 + 1 in 0 until h
            val in11 = x0 + 1 in 0 until w && y0 + 1 in 0 until h
            for (c in 0 until 3) {
                var v = 0f
                if (in00) v += w00 * (d[(y0 * w + x0) * 3 + c].toInt() and 0xFF)
                if (in10) v += w10 * (d[(y0 * w + x0 + 1) * 3 + c].toInt() and 0xFF)
                if (in01) v += w01 * (d[((y0 + 1) * w + x0) * 3 + c].toInt() and 0xFF)
                if (in11) v += w11 * (d[((y0 + 1) * w + x0 + 1) * 3 + c].toInt() and 0xFF)
                sink(c, x - roi.x0, y - roi.y0, v)
            }
        }
    }
}

/**
 * Warp a constant [value] square of size [size] through inv, into [roi].
 * Equivalent to `cv2.warpAffine(np.full((size, size), value), M, ...)`.
 */
fun warpConstantSquare(size: Int, value: Float, inv: Affine, roi: Rect): FloatArray {
    val out = FloatArray(roi.width * roi.height)
    for (y in roi.y0 until roi.y1) {
        for (x in roi.x0 until roi.x1) {
            val sx = inv.mapX(x.toDouble(), y.toDouble())
            val sy = inv.mapY(x.toDouble(), y.toDouble())
            val x0 = floor(sx).toInt(); val y0 = floor(sy).toInt()
            val fx = (sx - x0).toFloat(); val fy = (sy - y0).toFloat()
            var v = 0f
            if (x0 in 0 until size && y0 in 0 until size) v += (1 - fx) * (1 - fy)
            if (x0 + 1 in 0 until size && y0 in 0 until size) v += fx * (1 - fy)
            if (x0 in 0 until size && y0 + 1 in 0 until size) v += (1 - fx) * fy
            if (x0 + 1 in 0 until size && y0 + 1 in 0 until size) v += fx * fy
            out[(y - roi.y0) * roi.width + (x - roi.x0)] = v * value
        }
    }
    return out
}

/**
 * Min filter with a k x k window anchored like OpenCV's default (window
 * covers offsets -k/2 .. k-1-k/2). Pixels outside the array are ignored,
 * which matches `cv2.erode`'s default constant border.
 */
fun erode(src: FloatArray, w: Int, h: Int, k: Int): FloatArray {
    val lo = -(k / 2); val hi = k - 1 - k / 2
    val tmp = FloatArray(w * h)
    for (y in 0 until h) {
        for (x in 0 until w) {
            var m = Float.MAX_VALUE
            for (dx in maxOf(lo, -x)..minOf(hi, w - 1 - x)) m = minOf(m, src[y * w + x + dx])
            tmp[y * w + x] = m
        }
    }
    val out = FloatArray(w * h)
    for (y in 0 until h) {
        for (x in 0 until w) {
            var m = Float.MAX_VALUE
            for (dy in maxOf(lo, -y)..minOf(hi, h - 1 - y)) m = minOf(m, tmp[(y + dy) * w + x])
            out[y * w + x] = m
        }
    }
    return out
}

/** Port of `cv2.getGaussianKernel` for ksize > 7 (sigma <= 0 means auto). */
fun gaussianKernel(ksize: Int, sigma: Double = 0.0): FloatArray {
    require(ksize % 2 == 1)
    val s = if (sigma > 0) sigma else 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    val r = ksize / 2
    val k = DoubleArray(ksize) { i -> exp(-((i - r) * (i - r)) / (2 * s * s)) }
    val sum = k.sum()
    return FloatArray(ksize) { (k[it] / sum).toFloat() }
}

/** Separable Gaussian blur with OpenCV's default BORDER_REFLECT_101. */
fun gaussianBlur(src: FloatArray, w: Int, h: Int, ksize: Int): FloatArray {
    val kernel = gaussianKernel(ksize)
    val r = ksize / 2
    fun reflect(i: Int, n: Int): Int {
        if (n == 1) return 0
        var j = i
        while (j < 0 || j >= n) j = if (j < 0) -j else 2 * n - 2 - j
        return j
    }
    val tmp = FloatArray(w * h)
    for (y in 0 until h) for (x in 0 until w) {
        var s = 0f
        for (i in 0 until ksize) s += kernel[i] * src[y * w + reflect(x + i - r, w)]
        tmp[y * w + x] = s
    }
    val out = FloatArray(w * h)
    for (y in 0 until h) for (x in 0 until w) {
        var s = 0f
        for (i in 0 until ksize) s += kernel[i] * tmp[reflect(y + i - r, h) * w + x]
        out[y * w + x] = s
    }
    return out
}
