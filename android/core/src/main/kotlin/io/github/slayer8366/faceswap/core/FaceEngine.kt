package io.github.slayer8366.faceswap.core

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File

class NoFaceException(message: String) : Exception(message)

/** Paths to the three ONNX models plus the extracted swapper projection. */
class ModelFiles(val detector: File, val recognizer: File, val swapper: File, val emap: FloatArray)

/** Detection, recognition and swapping for single frames. Not thread-safe. */
class FaceEngine(models: ModelFiles, threads: Int = Runtime.getRuntime().availableProcessors()) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }
    private val sessions = ArrayList<OrtSession>()

    private fun open(file: File) = env.createSession(file.absolutePath, options).also { sessions.add(it) }

    val detector: ScrfdDetector
    val recognizer: ArcFace
    val swapper: InSwapper

    init {
        try {
            detector = ScrfdDetector(env, open(models.detector))
            recognizer = ArcFace(env, open(models.recognizer))
            swapper = InSwapper(env, open(models.swapper), models.emap)
        } catch (t: Throwable) {
            close()
            throw t
        }
    }

    fun detect(img: RgbImage): List<Face> = detector.detect(img)

    /**
     * Largest face in a still image, with its embedding. SCRFD often misses
     * a face that fills the frame, so a padded copy is tried as a fallback;
     * only the embedding is used afterwards, so coordinates don't matter.
     */
    fun faceFromStill(img: RgbImage, what: String = "image"): Face {
        var target = img
        var faces = detect(img)
        if (faces.isEmpty()) {
            target = img.pad(maxOf(img.width, img.height) / 2)
            faces = detect(target)
        }
        val face = faces.maxByOrNull { it.area } ?: throw NoFaceException("No face found in the $what.")
        face.embedding = recognizer.embed(target, face.kps)
        return face
    }

    /**
     * Swap the source identity ([latent], from [InSwapper.latent]) onto the
     * selected faces of [frame], in place. Returns how many were replaced.
     */
    fun swapFrame(
        frame: RgbImage,
        latent: FloatArray,
        mode: SwapMode,
        reference: FloatArray? = null,
        threshold: Float = 0.35f,
    ): Int {
        val faces = detect(frame)
        if (mode == SwapMode.REFERENCE) faces.forEach { it.embedding = recognizer.embed(frame, it.kps) }
        val targets = selectTargets(faces, mode, reference, threshold)
        targets.forEach { swapper.swap(frame, it.kps, latent) }
        return targets.size
    }

    override fun close() {
        sessions.forEach { it.close() }
        options.close()
    }
}
