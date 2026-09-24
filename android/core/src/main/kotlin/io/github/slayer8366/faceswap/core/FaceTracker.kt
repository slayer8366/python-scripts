package io.github.slayer8366.faceswap.core

import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Intersection over union of two (x1, y1, x2, y2) boxes. */
fun iou(a: FloatArray, b: FloatArray): Float {
    val w = max(0f, min(a[2], b[2]) - max(a[0], b[0]))
    val h = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
    val inter = w * h
    val union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return if (union <= 0f) 0f else inter / union
}

/**
 * Follows faces through a video, giving each one a stable track id.
 *
 * With [cropTracking] off, every frame gets a full detection and tracks are
 * only linked across frames. With it on, full detection runs every
 * [fullEvery] frames and each known face is re-detected in between inside a
 * crop around its last box, at the same pixel scale as the full detector.
 * Measured on desktop: about 11% faster per one-face frame with the int8
 * swapper, landmarks within 1.7 px of full detection (the detector's own
 * jitter is 1.1 px), but a face entering the shot can go unswapped for up to
 * [fullEvery] - 1 frames.
 */
class FaceTracker(
    /** Full-frame detection. */
    private val detectFull: (RgbImage) -> List<Face>,
    /**
     * Detection inside [Rect] of the frame at the given square input size
     * (a multiple of 32), returning coordinates relative to the region.
     */
    private val detectCrop: (RgbImage, Rect, Int) -> List<Face>,
    private val fullEvery: Int = 6,
    /** Input size of the full-frame detector, used to match its pixel scale in crops. */
    private val fullInputSize: Int = 640,
    private val cropTracking: Boolean = true,
) {
    init {
        require(fullEvery >= 1)
    }

    class Track internal constructor(val id: Int, face: Face) {
        var face: Face = face
            internal set

        /** Cached "is this the reference person?" decision; null means not checked yet. */
        var isTarget: Boolean? = null

        /** Frame index when [isTarget] was last decided. */
        var checkedAt: Int = -1
    }

    /** Index of the frame most recently passed to [update]. */
    var frameIndex = -1
        private set

    var fullDetections = 0
        private set
    var cropDetections = 0
        private set

    private var tracks = emptyList<Track>()
    private var nextId = 0
    private var lostSinceFull = false

    /** Faces in [frame], keeping track identity across calls. */
    fun update(frame: RgbImage): List<Track> {
        frameIndex++
        val full = !cropTracking || frameIndex % fullEvery == 0 || lostSinceFull
        tracks = if (full) fullUpdate(frame) else cropUpdate(frame)
        return tracks
    }

    private fun fullUpdate(frame: RgbImage): List<Track> {
        fullDetections++
        lostSinceFull = false
        val faces = detectFull(frame)
        val unmatched = tracks.toMutableList()
        val out = ArrayList<Track>()
        for (face in faces) {
            val best = unmatched.maxByOrNull { iou(it.face.bbox, face.bbox) }
            val track = if (best != null && iou(best.face.bbox, face.bbox) >= 0.3f) {
                unmatched.remove(best)
                best.also { it.face = face }
            } else {
                Track(nextId++, face)
            }
            out.add(track)
        }
        return out
    }

    private fun cropUpdate(frame: RgbImage): List<Track> {
        val scale = fullInputSize.toFloat() / max(frame.width, frame.height)
        val kept = ArrayList<Track>()
        for (track in tracks) {
            val b = track.face.bbox
            // Pick the detector input first (a multiple of 32), then a square crop
            // that it resizes by exactly the full detector's scale, so both see
            // the face at the same pixel scale. Crops at the frame edge are
            // shifted inward rather than shrunk, to keep that scale.
            val size = (((max(b[2] - b[0], b[3] - b[1]) * 2f * scale).toInt() + 31) / 32 * 32).coerceIn(64, 320)
            val side = (size / scale).roundToInt()
            val cx = ((b[0] + b[2]) / 2).toInt(); val cy = ((b[1] + b[3]) / 2).toInt()
            val w = min(side, frame.width); val h = min(side, frame.height)
            val x0 = (cx - side / 2).coerceIn(0, frame.width - w)
            val y0 = (cy - side / 2).coerceIn(0, frame.height - h)
            if (w < 16 || h < 16) { lostSinceFull = true; continue }
            cropDetections++
            val found = detectCrop(frame, Rect(x0, y0, x0 + w, y0 + h), size).map { f ->
                Face(
                    floatArrayOf(f.bbox[0] + x0, f.bbox[1] + y0, f.bbox[2] + x0, f.bbox[3] + y0),
                    f.score,
                    FloatArray(10) { i -> f.kps[i] + if (i % 2 == 0) x0 else y0 },
                )
            }
            val match = found.maxByOrNull { iou(it.bbox, b) }
            if (match == null || iou(match.bbox, b) < 0.3f) {
                lostSinceFull = true
                continue
            }
            track.face = match
            kept.add(track)
        }
        // Two tracks that converged on one face: keep the higher-scoring one.
        val sorted = kept.sortedByDescending { it.face.score }
        val deduped = ArrayList<Track>()
        for (t in sorted) if (deduped.none { iou(it.face.bbox, t.face.bbox) > 0.5f }) deduped.add(t)
        return deduped.sortedBy { it.id }
    }
}

/**
 * Per-video swapping: links faces across frames so reference matches are
 * cached (re-checked every [recheckEvery] frames), optionally with crop tracking.
 */
class TrackingSwapper(
    private val engine: FaceEngine,
    private val latent: FloatArray,
    private val mode: SwapMode,
    private val reference: FloatArray?,
    private val threshold: Float,
    cropTracking: Boolean = false,
    fullEvery: Int = 6,
    /** Re-check each face's reference match this often, so a track mix-up between crossing people is short-lived. */
    private val recheckEvery: Int = 6,
) {
    val tracker = FaceTracker(
        cropTracking = cropTracking,
        detectFull = { engine.detect(it) },
        detectCrop = { img, r, size -> engine.detector.detect(img.crop(r.x0, r.y0, r.width, r.height), size) },
        fullEvery = fullEvery,
        fullInputSize = engine.detector.inputSize,
    )

    var embeddings = 0
        private set

    fun process(frame: RgbImage): Int {
        val tracks = tracker.update(frame)
        val targets = when (mode) {
            SwapMode.ALL -> tracks
            SwapMode.LARGEST -> listOfNotNull(tracks.maxByOrNull { it.face.area })
            SwapMode.REFERENCE -> {
                val ref = requireNotNull(reference) { "Reference mode needs a reference embedding" }
                tracks.filter { t ->
                    val cached = t.isTarget
                    if (cached != null && tracker.frameIndex - t.checkedAt < recheckEvery) {
                        cached
                    } else {
                        embeddings++
                        t.checkedAt = tracker.frameIndex
                        (cosineSimilarity(engine.recognizer.embed(frame, t.face.kps), ref) >= threshold)
                            .also { t.isTarget = it }
                    }
                }
            }
        }
        targets.forEach { engine.swapper.swap(frame, it.face.kps, latent) }
        return targets.size
    }
}
