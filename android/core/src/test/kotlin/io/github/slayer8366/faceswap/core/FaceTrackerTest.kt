package io.github.slayer8366.faceswap.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Tracker logic against a scripted world of moving boxes; no models needed. */
class FaceTrackerTest {
    private fun face(x: Float, y: Float, s: Float = 40f, score: Float = 0.9f) =
        Face(floatArrayOf(x, y, x + s, y + s), score, FloatArray(10) { i -> if (i % 2 == 0) x + s / 2 else y + s / 2 })

    /** Faces at frame i come from [at]; the crop detector sees only faces fully inside its region. */
    private class World(val at: (Int) -> List<Face>) {
        var frame = 0
        val fullCalls = ArrayList<Int>()
        val cropSizes = ArrayList<Int>()
        val cropRegions = ArrayList<Rect>()

        fun tracker(fullEvery: Int) = FaceTracker(
            detectFull = { fullCalls.add(frame); at(frame) },
            detectCrop = { _, r, size ->
                cropSizes.add(size)
                cropRegions.add(r)
                at(frame).filter { f ->
                    f.bbox[0] >= r.x0 && f.bbox[1] >= r.y0 && f.bbox[2] <= r.x1 && f.bbox[3] <= r.y1
                }.map { f ->
                    Face(floatArrayOf(f.bbox[0] - r.x0, f.bbox[1] - r.y0, f.bbox[2] - r.x0, f.bbox[3] - r.y0), f.score,
                        FloatArray(10) { i -> f.kps[i] - if (i % 2 == 0) r.x0 else r.y0 })
                }
            },
            fullEvery = fullEvery,
        )

        /** Per-frame snapshots: tracks are updated in place, so copy what each frame saw. */
        class Seen(val id: Int, val bbox: FloatArray, val kps: FloatArray)

        fun run(t: FaceTracker, frames: Int, w: Int = 640, h: Int = 480): List<List<Seen>> {
            val img = RgbImage(w, h)
            return (0 until frames).map { i ->
                frame = i
                t.update(img).map { Seen(it.id, it.face.bbox.copyOf(), it.face.kps.copyOf()) }
            }
        }
    }

    @Test
    fun fullDetectionRunsOnScheduleAndIdsStayStable() {
        // One face drifting right 3 px per frame.
        val world = World { i -> listOf(face(100f + 3 * i, 100f)) }
        val t = world.tracker(fullEvery = 5)
        val frames = world.run(t, 12)
        frames.forEachIndexed { i, tracks ->
            assertEquals(1, tracks.size, "frame $i")
            assertEquals(100f + 3 * i, tracks[0].bbox[0], 0.01f, "frame $i x")
            assertEquals(120f + 3 * i, tracks[0].kps[0], 0.01f, "frame $i landmark x")
        }
        assertEquals(listOf(0, 5, 10), world.fullCalls)
        assertEquals(1, frames.flatten().map { it.id }.toSet().size, "track id changed")
        assertEquals(9, t.cropDetections)
    }

    @Test
    fun lostFaceTriggersFullDetectionNextFrame() {
        val world = World { i -> if (i < 3) listOf(face(200f, 200f)) else emptyList() }
        val t = world.tracker(fullEvery = 100)
        assertEquals(listOf(1, 1, 1, 0, 0, 0), world.run(t, 6).map { it.size })
        // Frame 3 lost it in the crop, so frame 4 re-detected the whole frame;
        // with no tracks left, it then waits for the schedule.
        assertEquals(listOf(0, 4), world.fullCalls)
    }

    @Test
    fun newFaceIsPickedUpAtNextFullDetectionThenTracked() {
        val world = World { i -> listOf(face(100f, 100f)) + if (i >= 2) listOf(face(400f, 300f)) else emptyList() }
        val t = world.tracker(fullEvery = 4)
        val frames = world.run(t, 7)
        assertEquals(listOf(1, 1, 1, 1, 2, 2, 2), frames.map { it.size })
        assertEquals(listOf(0, 4), world.fullCalls)
        assertEquals(frames[4].map { it.id }, frames[6].map { it.id })
    }

    @Test
    fun convergedTracksAreDeduplicated() {
        // Two overlapping detections at frame 0 (as if NMS let both through), one face after.
        val world = World { i -> if (i == 0) listOf(face(100f, 100f), face(104f, 100f, score = 0.8f)) else listOf(face(100f, 100f)) }
        val t = world.tracker(fullEvery = 10)
        assertEquals(listOf(2, 1), world.run(t, 2).map { it.size })
    }

    @Test
    fun cropInputMatchesFullDetectionScale() {
        val world = World { listOf(face(100f, 100f, s = 100f)) }
        world.run(world.tracker(fullEvery = 10), 2, w = 1280, h = 720)
        // 2x the 100 px box at the full detector's scale (640/1280) is 100 px, rounded up to 128;
        // the crop is then 128 / 0.5 = 256 px, so the detector resizes it by exactly 0.5.
        assertEquals(listOf(128), world.cropSizes)
        assertEquals(listOf(256 to 256), world.cropRegions.map { it.width to it.height })
    }

    @Test
    fun iouBasics() {
        assertEquals(1f, iou(floatArrayOf(0f, 0f, 10f, 10f), floatArrayOf(0f, 0f, 10f, 10f)), 1e-6f)
        assertEquals(0f, iou(floatArrayOf(0f, 0f, 10f, 10f), floatArrayOf(20f, 20f, 30f, 30f)), 1e-6f)
        assertTrue(iou(floatArrayOf(0f, 0f, 10f, 10f), floatArrayOf(5f, 0f, 15f, 10f)) in 0.33f..0.34f)
    }
}
