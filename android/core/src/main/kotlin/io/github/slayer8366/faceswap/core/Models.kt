package io.github.slayer8366.faceswap.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.FloatBuffer
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** Fill a 1x3xHxW float tensor (RGB order) with (pixel - mean) / std. */
internal fun toChwTensor(img: RgbImage, mean: Float, std: Float): FloatBuffer {
    val plane = img.width * img.height
    val out = FloatArray(plane * 3)
    for (i in 0 until plane) {
        for (c in 0 until 3) out[c * plane + i] = ((img.data[i * 3 + c].toInt() and 0xFF) - mean) / std
    }
    return FloatBuffer.wrap(out)
}

internal fun runSingle(
    env: OrtEnvironment,
    session: OrtSession,
    inputs: Map<String, Pair<FloatBuffer, LongArray>>,
): List<Pair<LongArray, FloatArray>> {
    val tensors = inputs.mapValues { (_, v) -> OnnxTensor.createTensor(env, v.first, v.second) }
    try {
        session.run(tensors).use { result ->
            return result.map { entry ->
                val t = entry.value as OnnxTensor
                val fb = t.floatBuffer
                t.info.shape to FloatArray(fb.remaining()).also { fb.get(it) }
            }
        }
    } finally {
        tensors.values.forEach { it.close() }
    }
}

/** SCRFD face detector (buffalo_l `det_10g.onnx`). Port of insightface `SCRFD.detect`. */
class ScrfdDetector(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    val inputSize: Int = 640,
    val threshold: Float = 0.5f,
    val nmsThreshold: Float = 0.4f,
) {
    private val inputName = session.inputNames.first()
    private val numAnchors = 2

    fun detect(img: RgbImage): List<Face> {
        val imRatio = img.height.toDouble() / img.width
        val (newW, newH) = if (imRatio > 1.0) {
            (inputSize / imRatio).toInt() to inputSize
        } else {
            inputSize to (inputSize * imRatio).toInt()
        }
        val detScale = newH.toFloat() / img.height
        val canvas = RgbImage(inputSize, inputSize)
        val resized = img.resize(maxOf(1, newW), maxOf(1, newH))
        for (row in 0 until resized.height) {
            System.arraycopy(resized.data, row * resized.width * 3, canvas.data, row * inputSize * 3, resized.width * 3)
        }
        val outputs = runSingle(
            env, session,
            mapOf(inputName to (toChwTensor(canvas, 127.5f, 128f) to longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()))),
        )
        // Match outputs by shape rather than name/order: last dim 1 = scores,
        // 4 = boxes, 10 = landmarks; anchor count gives the stride.
        fun group(lastDim: Long) = outputs.filter { it.first.last() == lastDim }.sortedByDescending { it.first[0] }
        val scores = group(1); val boxes = group(4); val kpss = group(10)
        require(scores.size == 3 && boxes.size == 3 && kpss.size == 3) { "Unexpected SCRFD outputs" }

        val candidates = ArrayList<Face>()
        for (level in 0 until 3) {
            val n = scores[level].first[0].toInt()
            val fw = sqrt(n / numAnchors.toDouble()).roundToInt()
            val stride = inputSize / fw
            val s = scores[level].second; val b = boxes[level].second; val k = kpss[level].second
            for (i in 0 until n) {
                if (s[i] < threshold) continue
                val cell = i / numAnchors
                val cx = (cell % fw) * stride.toFloat()
                val cy = (cell / fw) * stride.toFloat()
                val bbox = floatArrayOf(
                    (cx - b[i * 4] * stride) / detScale, (cy - b[i * 4 + 1] * stride) / detScale,
                    (cx + b[i * 4 + 2] * stride) / detScale, (cy + b[i * 4 + 3] * stride) / detScale,
                )
                val kps = FloatArray(10) { j ->
                    ((if (j % 2 == 0) cx else cy) + k[i * 10 + j] * stride) / detScale
                }
                candidates.add(Face(bbox, s[i], kps))
            }
        }
        return nms(candidates.sortedByDescending { it.score }, nmsThreshold)
    }
}

/** ArcFace recogniser (buffalo_l `w600k_r50.onnx`). Returns L2-normalised embeddings. */
class ArcFace(private val env: OrtEnvironment, private val session: OrtSession) {
    private val inputName = session.inputNames.first()

    fun embed(img: RgbImage, kps: FloatArray): FloatArray {
        val crop = warpAffine(img, estimateNorm(kps, 112), 112, 112)
        val out = runSingle(env, session, mapOf(inputName to (toChwTensor(crop, 127.5f, 127.5f) to longArrayOf(1, 3, 112, 112))))
        return l2Normalize(out.single().second)
    }
}

/**
 * inswapper_128 face swapper. Port of insightface `INSwapper.get` with
 * `paste_back=True`, computed only over the region the face covers.
 */
class InSwapper(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val emap: FloatArray,
) {
    private val size = 128
    private val targetName: String
    private val sourceName: String

    init {
        require(emap.size == 512 * 512) { "emap must be 512x512" }
        val names = session.inputInfo.entries.associate { (name, info) ->
            name to ((info.info as TensorInfo).shape.size)
        }
        targetName = names.entries.first { it.value == 4 }.key
        sourceName = names.entries.first { it.value == 2 }.key
    }

    /** Project a normalised ArcFace embedding into the swapper's latent space. */
    fun latent(sourceEmbedding: FloatArray): FloatArray {
        val out = FloatArray(512)
        for (i in 0 until 512) {
            val e = sourceEmbedding[i]
            if (e == 0f) continue
            for (j in 0 until 512) out[j] += e * emap[i * 512 + j]
        }
        return l2Normalize(out)
    }

    /** Generate the 128x128 swapped face for [kps] in [frame] (paste_back=False). */
    fun generate(frame: RgbImage, kps: FloatArray, latent: FloatArray): Pair<RgbImage, Affine> {
        val m = estimateNorm(kps, size)
        val aligned = warpAffine(frame, m, size, size)
        val out = runSingle(
            env, session,
            mapOf(
                targetName to (toChwTensor(aligned, 0f, 255f) to longArrayOf(1, 3, size.toLong(), size.toLong())),
                sourceName to (FloatBuffer.wrap(latent) to longArrayOf(1, 512)),
            ),
        ).single().second
        val plane = size * size
        val fake = RgbImage(size, size)
        for (i in 0 until plane) {
            for (c in 0 until 3) fake.data[i * 3 + c] = (out[c * plane + i] * 255f).coerceIn(0f, 255f).toInt().toByte()
        }
        return fake to m
    }

    /** Swap the face at [kps] in place. */
    fun swap(frame: RgbImage, kps: FloatArray, latent: FloatArray) {
        val (fake, m) = generate(frame, kps, latent)
        pasteBack(frame, fake, m)
    }

    internal fun pasteBack(frame: RgbImage, fake: RgbImage, m: Affine) {
        val footprint = mappedBounds(m.invert(), size, size)
        val margin = maxOf(footprint.width, footprint.height) / 20 + 8
        val roi = footprint.expand(margin, frame.width, frame.height)
        if (roi.isEmpty) return
        val rw = roi.width; val rh = roi.height

        val fakeWarped = ByteArray(rw * rh * 3)
        warpInto(fake, m, roi) { c, x, y, v -> fakeWarped[(y * rw + x) * 3 + c] = v.roundToInt().toByte() }

        val white = warpConstantSquare(size, 255f, m, roi)
        var minX = Int.MAX_VALUE; var maxX = -1; var minY = Int.MAX_VALUE; var maxY = -1
        for (y in 0 until rh) for (x in 0 until rw) {
            val i = y * rw + x
            if (white[i] > 20f) {
                white[i] = 255f
                if (x < minX) minX = x; if (x > maxX) maxX = x
                if (y < minY) minY = y; if (y > maxY) maxY = y
            }
        }
        if (maxX < 0) return
        val maskSize = sqrt(((maxY - minY) * (maxX - minX)).toDouble()).toInt()
        val eroded = erode(white, rw, rh, maxOf(maskSize / 10, 10))
        val k = maxOf(maskSize / 20, 5)
        val mask = gaussianBlur(eroded, rw, rh, 2 * k + 1)

        for (y in 0 until rh) for (x in 0 until rw) {
            val a = mask[y * rw + x] / 255f
            if (a <= 0f) continue
            val fi = ((roi.y0 + y) * frame.width + roi.x0 + x) * 3
            val wi = (y * rw + x) * 3
            for (c in 0 until 3) {
                val t = frame.data[fi + c].toInt() and 0xFF
                val f = fakeWarped[wi + c].toInt() and 0xFF
                frame.data[fi + c] = (a * f + (1 - a) * t).toInt().toByte()
            }
        }
    }
}
