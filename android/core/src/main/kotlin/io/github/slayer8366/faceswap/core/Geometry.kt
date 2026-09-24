package io.github.slayer8366.faceswap.core

import kotlin.math.max
import kotlin.math.min

/** 2x3 affine matrix mapping (x, y) to (a*x + b*y + c, d*x + e*y + f). */
data class Affine(val a: Double, val b: Double, val c: Double, val d: Double, val e: Double, val f: Double) {
    fun mapX(x: Double, y: Double) = a * x + b * y + c
    fun mapY(x: Double, y: Double) = d * x + e * y + f

    fun invert(): Affine {
        val det = a * e - b * d
        require(det != 0.0) { "Affine matrix is singular" }
        val ia = e / det
        val ib = -b / det
        val id = -d / det
        val ie = a / det
        return Affine(ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f))
    }

    fun toArray() = doubleArrayOf(a, b, c, d, e, f)
}

/** Axis-aligned integer rectangle, [x0, x1) by [y0, y1). */
data class Rect(val x0: Int, val y0: Int, val x1: Int, val y1: Int) {
    val width get() = x1 - x0
    val height get() = y1 - y0
    val isEmpty get() = width <= 0 || height <= 0
}

/** ArcFace's canonical 5-point landmark positions in a 112x112 crop. */
val ARCFACE_DST = doubleArrayOf(
    38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041,
)

/**
 * Least-squares similarity transform (rotation, uniform scale, translation)
 * mapping points [src] onto [dst]; both are flat x,y arrays. This is the
 * same optimum scikit-image's `SimilarityTransform.estimate` (Umeyama)
 * finds, written in closed form for the 2-D case.
 */
fun similarityTransform(src: FloatArray, dst: DoubleArray): Affine {
    require(src.size == dst.size && src.size >= 4 && src.size % 2 == 0)
    val n = src.size / 2
    var sx = 0.0; var sy = 0.0; var dx = 0.0; var dy = 0.0
    for (i in 0 until n) {
        sx += src[2 * i]; sy += src[2 * i + 1]; dx += dst[2 * i]; dy += dst[2 * i + 1]
    }
    sx /= n; sy /= n; dx /= n; dy /= n
    var num1 = 0.0; var num2 = 0.0; var den = 0.0
    for (i in 0 until n) {
        val x = src[2 * i] - sx; val y = src[2 * i + 1] - sy
        val u = dst[2 * i] - dx; val v = dst[2 * i + 1] - dy
        num1 += x * u + y * v
        num2 += x * v - y * u
        den += x * x + y * y
    }
    require(den > 0.0) { "Source points are degenerate" }
    val p = num1 / den
    val q = num2 / den
    return Affine(p, -q, dx - (p * sx - q * sy), q, p, dy - (q * sx + p * sy))
}

/** Port of insightface `face_align.estimate_norm`. */
fun estimateNorm(kps: FloatArray, imageSize: Int): Affine {
    require(kps.size == 10)
    val (ratio, diffX) = when {
        imageSize % 112 == 0 -> imageSize / 112.0 to 0.0
        imageSize % 128 == 0 -> imageSize / 128.0 to 8.0 * imageSize / 128.0
        else -> throw IllegalArgumentException("imageSize must be a multiple of 112 or 128")
    }
    val dst = DoubleArray(10) { i -> ARCFACE_DST[i] * ratio + if (i % 2 == 0) diffX else 0.0 }
    return similarityTransform(kps, dst)
}

/** Integer bounding box of a w x h rectangle after mapping through [m]. */
fun mappedBounds(m: Affine, w: Int, h: Int): Rect {
    var minX = Double.MAX_VALUE; var minY = Double.MAX_VALUE
    var maxX = -Double.MAX_VALUE; var maxY = -Double.MAX_VALUE
    for ((x, y) in listOf(0.0 to 0.0, w - 1.0 to 0.0, 0.0 to h - 1.0, w - 1.0 to h - 1.0)) {
        val mx = m.mapX(x, y); val my = m.mapY(x, y)
        minX = min(minX, mx); maxX = max(maxX, mx); minY = min(minY, my); maxY = max(maxY, my)
    }
    return Rect(
        kotlin.math.floor(minX).toInt(), kotlin.math.floor(minY).toInt(),
        kotlin.math.ceil(maxX).toInt() + 1, kotlin.math.ceil(maxY).toInt() + 1,
    )
}

fun Rect.expand(margin: Int, maxW: Int, maxH: Int) =
    Rect(max(0, x0 - margin), max(0, y0 - margin), min(maxW, x1 + margin), min(maxH, y1 + margin))
