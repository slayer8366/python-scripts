package io.github.slayer8366.faceswap.core

import kotlin.math.max
import kotlin.math.sqrt

/** One detected face: box (x1, y1, x2, y2), score, and 5 landmarks as x,y pairs. */
class Face(val bbox: FloatArray, val score: Float, val kps: FloatArray) {
    /** L2-normalised ArcFace embedding, filled in only when needed. */
    var embedding: FloatArray? = null

    val area: Float get() = max(0f, bbox[2] - bbox[0]) * max(0f, bbox[3] - bbox[1])
}

enum class SwapMode {
    /** Every detected face. */
    ALL,

    /** Only the biggest face in each frame. */
    LARGEST,

    /** Faces whose embedding matches a reference photo. */
    REFERENCE,
}

fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
    require(a.size == b.size)
    var dot = 0.0; var na = 0.0; var nb = 0.0
    for (i in a.indices) {
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
    }
    val denom = sqrt(na) * sqrt(nb)
    return if (denom == 0.0) 0f else (dot / denom).toFloat()
}

fun l2Normalize(v: FloatArray): FloatArray {
    var n = 0.0
    for (x in v) n += x * x
    n = sqrt(n)
    return if (n == 0.0) v.copyOf() else FloatArray(v.size) { (v[it] / n).toFloat() }
}

/**
 * Pick which faces get replaced. Mirrors `select_targets` in the Python app.
 * In [SwapMode.REFERENCE] every face must already carry an embedding.
 */
fun selectTargets(
    faces: List<Face>,
    mode: SwapMode,
    reference: FloatArray? = null,
    threshold: Float = 0.35f,
): List<Face> {
    if (faces.isEmpty()) return emptyList()
    return when (mode) {
        SwapMode.ALL -> faces
        SwapMode.LARGEST -> listOf(faces.maxBy { it.area })
        SwapMode.REFERENCE -> {
            requireNotNull(reference) { "Reference mode needs a reference embedding" }
            faces.filter { f -> f.embedding?.let { cosineSimilarity(it, reference) >= threshold } == true }
        }
    }
}

/** Greedy NMS over faces already sorted by descending score (insightface convention, +1 areas). */
fun nms(sorted: List<Face>, threshold: Float): List<Face> {
    val keep = ArrayList<Face>()
    val suppressed = BooleanArray(sorted.size)
    for (i in sorted.indices) {
        if (suppressed[i]) continue
        val a = sorted[i]
        keep.add(a)
        val areaA = (a.bbox[2] - a.bbox[0] + 1) * (a.bbox[3] - a.bbox[1] + 1)
        for (j in i + 1 until sorted.size) {
            if (suppressed[j]) continue
            val b = sorted[j]
            val w = max(0f, minOf(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0]) + 1)
            val h = max(0f, minOf(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1]) + 1)
            val inter = w * h
            val areaB = (b.bbox[2] - b.bbox[0] + 1) * (b.bbox[3] - b.bbox[1] + 1)
            if (inter / (areaA + areaB - inter) > threshold) suppressed[j] = true
        }
    }
    return keep
}
