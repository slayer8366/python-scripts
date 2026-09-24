package io.github.slayer8366.faceswap

import android.content.Context
import android.graphics.ImageFormat
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import io.github.slayer8366.faceswap.core.Plane
import io.github.slayer8366.faceswap.core.RgbImage
import io.github.slayer8366.faceswap.core.YuvMatrix
import io.github.slayer8366.faceswap.core.rgbToYuv420
import io.github.slayer8366.faceswap.core.yuv420ToRgb
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.CancellationException

/** Edits one upright frame in place and returns how many faces it replaced. */
fun interface FrameProcessor {
    fun process(frame: RgbImage): Int
}

class SwapStats(var frames: Int = 0, var framesWithSwap: Int = 0, var faces: Int = 0, var audio: AudioResult = AudioResult.NONE)

enum class AudioResult { NONE, COPIED, UNSUPPORTED }

class UnsupportedVideoException(message: String) : Exception(message)

/**
 * Decodes video with MediaCodec, hands each frame to a [FrameProcessor],
 * and encodes H.264 into an MP4, copying the original audio track.
 *
 * Frames are rotated upright before processing (the detector expects
 * upright faces), so the output carries no rotation flag. Presentation
 * timestamps are kept, so variable-frame-rate phone video stays in sync.
 */
class VideoTranscoder(private val context: Context) {
    private val timeoutUs = 10_000L

    fun run(
        input: Uri,
        output: File,
        processor: FrameProcessor,
        label: String?,
        /** Stop after this much video, for quick previews. */
        maxDurationUs: Long?,
        progress: (Float, SwapStats) -> Unit,
        cancelled: () -> Boolean,
    ): SwapStats {
        val stats = SwapStats()
        var encoder: Encoder? = null
        var firstPtsUs = -1L
        var lastPtsUs = 0L
        try {
            decodeFrames(input, 0L, maxDurationUs, cancelled) { video, upright, ptsUs ->
                val enc = encoder ?: Encoder(upright.width, upright.height, video.fps, output).also {
                    encoder = it
                    it.prepareAudio(input)
                }
                val fitted = enc.fit(upright)
                val n = processor.process(fitted)
                enc.label(label)?.drawOn(fitted)
                enc.encode(fitted, ptsUs)
                stats.frames++
                stats.faces += n
                if (n > 0) stats.framesWithSwap++
                if (firstPtsUs < 0) firstPtsUs = ptsUs
                lastPtsUs = ptsUs
                val endUs = maxDurationUs?.let { if (video.durationUs > 0) minOf(it, video.durationUs) else it } ?: video.durationUs
                val span = endUs - firstPtsUs
                progress(if (span > 0) ((lastPtsUs - firstPtsUs).toFloat() / span).coerceIn(0f, 1f) else 0f, stats)
                true
            }
            val enc = encoder ?: throw UnsupportedVideoException("No frames could be decoded.")
            enc.finish(cancelled)
            stats.audio = enc.copyAudio(lastPtsUs, maxDurationUs)
            enc.close()
            encoder = null
        } finally {
            encoder?.abort()
        }
        return stats
    }

    /**
     * The first frame at or after [timeUs], upright, decoded exactly as
     * [run] decodes it, so a preview matches the final output.
     */
    fun frameAt(input: Uri, timeUs: Long): RgbImage {
        var result: RgbImage? = null
        decodeFrames(input, timeUs, null, { false }) { _, upright, ptsUs ->
            if (ptsUs >= timeUs) result = upright
            result == null
        }
        return result ?: throw UnsupportedVideoException("No frame found at ${timeUs / 1000} ms.")
    }

    class VideoInfo(val durationUs: Long, val fps: Int, val rotation: Int)

    /**
     * Decode from the sync frame before [startUs] up to [maxUs], passing each
     * upright frame and its timestamp to [onFrame] until it returns false.
     */
    private fun decodeFrames(
        input: Uri,
        startUs: Long,
        maxUs: Long?,
        cancelled: () -> Boolean,
        onFrame: (VideoInfo, RgbImage, Long) -> Boolean,
    ): VideoInfo {
        val extractor = MediaExtractor().apply { setDataSource(context, input, null) }
        try {
            val videoTrack = (0 until extractor.trackCount).firstOrNull { mime(extractor.getTrackFormat(it)).startsWith("video/") }
                ?: throw UnsupportedVideoException("No video track found.")
            val inFormat = extractor.getTrackFormat(videoTrack)
            extractor.selectTrack(videoTrack)
            if (startUs > 0) extractor.seekTo(startUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
            val video = VideoInfo(
                durationUs = inFormat.longOr(MediaFormat.KEY_DURATION, 0L),
                fps = inFormat.intOr(MediaFormat.KEY_FRAME_RATE, 30).coerceIn(1, 240),
                rotation = inFormat.intOr(MediaFormat.KEY_ROTATION, 0),
            )

            val decoder = MediaCodec.createDecoderByType(mime(inFormat))
            try {
                inFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                decoder.configure(inFormat, null, null, 0)
                decoder.start()
                var matrix = defaultMatrix(inFormat.intOr(MediaFormat.KEY_HEIGHT, 1080))
                val info = MediaCodec.BufferInfo()
                var inputDone = false
                while (true) {
                    if (cancelled()) throw CancellationException("Cancelled")
                    if (!inputDone) {
                        val idx = decoder.dequeueInputBuffer(timeoutUs)
                        if (idx >= 0) {
                            val buf = decoder.getInputBuffer(idx)!!
                            val n = extractor.readSampleData(buf, 0)
                            val t = extractor.sampleTime
                            if (n < 0 || (maxUs != null && t > maxUs)) {
                                decoder.queueInputBuffer(idx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                decoder.queueInputBuffer(idx, 0, n, t, 0)
                                extractor.advance()
                            }
                        }
                    }
                    val idx = decoder.dequeueOutputBuffer(info, timeoutUs)
                    if (idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        matrix = matrixFor(decoder.outputFormat) ?: matrix
                    } else if (idx >= 0) {
                        val eos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        val pts = info.presentationTimeUs
                        val wanted = info.size > 0 && (maxUs == null || pts <= maxUs)
                        val frame = if (wanted) {
                            decoder.getOutputImage(idx)?.use { readFrame(it, matrix) }
                                ?: throw UnsupportedVideoException("This phone's decoder doesn't expose readable frames for this video.")
                        } else {
                            null
                        }
                        decoder.releaseOutputBuffer(idx, false)
                        if (frame != null && !onFrame(video, frame.rotate(video.rotation), pts)) return video
                        if (eos) return video
                    }
                }
            } finally {
                runCatching { decoder.stop() }
                decoder.release()
            }
        } finally {
            extractor.release()
        }
    }

    /** Copy the decoder's output image (cropped) into RGB. */
    private fun readFrame(image: Image, matrix: YuvMatrix): RgbImage {
        if (image.format != ImageFormat.YUV_420_888) {
            throw UnsupportedVideoException(
                "Unsupported frame format ${image.format}. HDR / 10-bit video isn't supported yet; " +
                    "record in SDR or convert the clip first.",
            )
        }
        val crop = image.cropRect
        val p = image.planes
        val out = RgbImage(crop.width(), crop.height())
        yuv420ToRgb(
            Plane(p[0].buffer, p[0].rowStride, p[0].pixelStride),
            Plane(p[1].buffer, p[1].rowStride, p[1].pixelStride),
            Plane(p[2].buffer, p[2].rowStride, p[2].pixelStride),
            matrix, out, crop.left, crop.top,
        )
        return out
    }

    /** H.264 encoder plus muxer, configured lazily from the first upright frame. */
    private inner class Encoder(srcW: Int, srcH: Int, private val fps: Int, private val output: File) {
        private val codec: MediaCodec
        val width: Int
        val height: Int
        private val matrix: YuvMatrix
        private val muxer = MediaMuxer(output.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        private var videoTrack = -1
        private var audioTrack = -1
        private var audioFormat: MediaFormat? = null
        private var muxerStarted = false
        private var labelCache: Pair<String, LabelOverlay>? = null
        private val info = MediaCodec.BufferInfo()

        init {
            val c = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            try {
                val caps = c.codecInfo.getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_AVC).videoCapabilities
                    ?: throw UnsupportedVideoException("No H.264 video encoder on this phone.")
                var w = srcW and 1.inv()
                var h = srcH and 1.inv()
                // Shrink until the encoder accepts the size (some can't do 4K or portrait 1080p).
                while (!caps.isSizeSupported(w, h)) {
                    if (w <= 320 || h <= 320) throw UnsupportedVideoException("This phone can't encode ${srcW}x$srcH video.")
                    w = ((w * 3 / 4) / caps.widthAlignment * caps.widthAlignment) and 1.inv()
                    h = ((h * 3 / 4) / caps.heightAlignment * caps.heightAlignment) and 1.inv()
                }
                width = w
                height = h
                matrix = defaultMatrix(h)
                val bitrate = caps.bitrateRange.clamp((w.toLong() * h * fps * 0.15).toInt())
                val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, w, h).apply {
                    setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                    setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
                    setInteger(MediaFormat.KEY_FRAME_RATE, fps)
                    setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
                    setInteger(MediaFormat.KEY_COLOR_STANDARD,
                        if (matrix === YuvMatrix.BT709) MediaFormat.COLOR_STANDARD_BT709 else MediaFormat.COLOR_STANDARD_BT601_NTSC)
                    setInteger(MediaFormat.KEY_COLOR_RANGE, MediaFormat.COLOR_RANGE_LIMITED)
                    setInteger(MediaFormat.KEY_COLOR_TRANSFER, MediaFormat.COLOR_TRANSFER_SDR_VIDEO)
                }
                c.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                c.start()
            } catch (t: Throwable) {
                c.release()
                muxer.release()
                throw t
            }
            codec = c
        }

        /** Crop odd edges or scale so the frame matches the encoder size. */
        fun fit(img: RgbImage): RgbImage = when {
            img.width == width && img.height == height -> img
            img.width - width in 0..1 && img.height - height in 0..1 -> img.crop(0, 0, width, height)
            else -> img.resize(width, height)
        }

        fun label(text: String?): LabelOverlay? {
            if (text == null) return null
            labelCache?.let { (t, overlay) -> if (t == text) return overlay }
            return LabelOverlay(text, width, height).also { labelCache = text to it }
        }

        fun encode(frame: RgbImage, ptsUs: Long) {
            while (true) {
                val idx = codec.dequeueInputBuffer(timeoutUs)
                if (idx >= 0) {
                    val image = codec.getInputImage(idx)
                        ?: throw UnsupportedVideoException("This phone's encoder doesn't accept YUV images.")
                    val p = image.planes
                    rgbToYuv420(
                        frame,
                        Plane(p[0].buffer, p[0].rowStride, p[0].pixelStride),
                        Plane(p[1].buffer, p[1].rowStride, p[1].pixelStride),
                        Plane(p[2].buffer, p[2].rowStride, p[2].pixelStride),
                        matrix,
                    )
                    codec.queueInputBuffer(idx, 0, width * height * 3 / 2, ptsUs, 0)
                    break
                }
                drain(false)
            }
            drain(false)
        }

        fun finish(cancelled: () -> Boolean) {
            while (true) {
                val idx = codec.dequeueInputBuffer(timeoutUs)
                if (idx >= 0) {
                    codec.queueInputBuffer(idx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    break
                }
                drain(false)
            }
            while (!drain(true)) {
                if (cancelled()) throw CancellationException("Cancelled")
            }
        }

        /** Returns true once the encoder has emitted end-of-stream. */
        private fun drain(waitForEos: Boolean): Boolean {
            while (true) {
                val idx = codec.dequeueOutputBuffer(info, if (waitForEos) timeoutUs else 0)
                when {
                    idx == MediaCodec.INFO_TRY_AGAIN_LATER -> return false
                    idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> startMuxer(codec.outputFormat)
                    idx >= 0 -> {
                        val buf = codec.getOutputBuffer(idx)!!
                        val config = info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0
                        if (!config && info.size > 0) {
                            check(muxerStarted) { "Encoder produced data before its format" }
                            muxer.writeSampleData(videoTrack, buf, info)
                        }
                        codec.releaseOutputBuffer(idx, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return true
                    }
                }
            }
        }

        private fun startMuxer(videoFormat: MediaFormat) {
            check(!muxerStarted) { "Encoder format changed twice" }
            videoTrack = muxer.addTrack(videoFormat)
            // Every track must be added before start(), so the audio track
            // is registered now and its samples are copied after the video.
            audioFormat = findAudioFormat()
            audioFormat?.let { f ->
                audioTrack = try { muxer.addTrack(f) } catch (e: Exception) { -1 }
            }
            muxer.start()
            muxerStarted = true
        }

        private var audioSource: Pair<MediaExtractor, Int>? = null

        private fun findAudioFormat(): MediaFormat? = audioSource?.let { (ex, t) -> ex.getTrackFormat(t) }

        /** Must be called before encoding starts so the muxer can register audio. */
        fun prepareAudio(input: Uri) {
            val ex = MediaExtractor().apply { setDataSource(context, input, null) }
            val track = (0 until ex.trackCount).firstOrNull { mime(ex.getTrackFormat(it)).startsWith("audio/") }
            if (track == null) ex.release() else audioSource = ex to track
        }

        fun copyAudio(lastVideoPtsUs: Long, maxUs: Long?): AudioResult {
            val (ex, track) = audioSource ?: return AudioResult.NONE
            if (audioTrack < 0) return AudioResult.UNSUPPORTED
            ex.selectTrack(track)
            val size = audioFormat?.intOr(MediaFormat.KEY_MAX_INPUT_SIZE, 0)?.takeIf { it > 0 } ?: (1 shl 20)
            val buf = ByteBuffer.allocateDirect(size)
            val info = MediaCodec.BufferInfo()
            val limit = minOf(lastVideoPtsUs, maxUs ?: Long.MAX_VALUE)
            while (true) {
                val n = ex.readSampleData(buf, 0)
                if (n < 0 || ex.sampleTime > limit) break
                info.set(0, n, ex.sampleTime,
                    if (ex.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0)
                muxer.writeSampleData(audioTrack, buf, info)
                ex.advance()
            }
            return AudioResult.COPIED
        }

        fun close() {
            codec.stop()
            codec.release()
            muxer.stop()
            muxer.release()
            audioSource?.first?.release()
        }

        fun abort() {
            runCatching { codec.stop() }
            codec.release()
            if (muxerStarted) runCatching { muxer.stop() }
            runCatching { muxer.release() }
            audioSource?.first?.release()
            output.delete()
        }
    }

    private fun mime(f: MediaFormat) = f.getString(MediaFormat.KEY_MIME) ?: ""

    /** Colour matrix the decoder reports, if it reports one. */
    private fun matrixFor(f: MediaFormat): YuvMatrix? {
        val full = f.intOr(MediaFormat.KEY_COLOR_RANGE, MediaFormat.COLOR_RANGE_LIMITED) == MediaFormat.COLOR_RANGE_FULL
        return when (f.intOr(MediaFormat.KEY_COLOR_STANDARD, -1)) {
            MediaFormat.COLOR_STANDARD_BT709 -> if (full) YuvMatrix(0.2126, 0.0722, true) else YuvMatrix.BT709
            MediaFormat.COLOR_STANDARD_BT601_NTSC, MediaFormat.COLOR_STANDARD_BT601_PAL ->
                if (full) YuvMatrix(0.299, 0.114, true) else YuvMatrix.BT601
            MediaFormat.COLOR_STANDARD_BT2020 -> YuvMatrix.BT2020
            else -> null
        }
    }

    /** Convention when a stream doesn't say: BT.709 for HD, BT.601 for SD. */
    private fun defaultMatrix(height: Int) = if (height >= 720) YuvMatrix.BT709 else YuvMatrix.BT601
}

private fun MediaFormat.intOr(key: String, default: Int) = if (containsKey(key)) getInteger(key) else default
private fun MediaFormat.longOr(key: String, default: Long) = if (containsKey(key)) getLong(key) else default
