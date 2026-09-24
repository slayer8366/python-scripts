package io.github.slayer8366.faceswap

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import io.github.slayer8366.faceswap.core.Plane
import io.github.slayer8366.faceswap.core.RgbImage
import io.github.slayer8366.faceswap.core.YuvMatrix
import io.github.slayer8366.faceswap.core.rgbToYuv420
import java.io.File
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.sin

/** Builds small H.264 + AAC test clips on the device, so no binary fixtures are needed. */
object ClipMaker {
    private class Sample(val data: ByteArray, val ptsUs: Long, val flags: Int)

    fun make(
        out: File,
        width: Int,
        height: Int,
        frames: Int,
        fps: Int,
        rotation: Int = 0,
        audio: Boolean = true,
        paint: (index: Int, img: RgbImage) -> Unit,
    ) {
        val (videoFormat, video) = encodeVideo(width, height, frames, fps, paint)
        val audioData = if (audio) encodeAudio(frames * 1_000_000L / fps) else null
        val muxer = MediaMuxer(out.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        try {
            muxer.setOrientationHint(rotation)
            val vt = muxer.addTrack(videoFormat)
            val at = audioData?.let { muxer.addTrack(it.first) }
            muxer.start()
            write(muxer, vt, video)
            if (at != null) write(muxer, at, audioData.second)
            muxer.stop()
        } finally {
            muxer.release()
        }
    }

    private fun write(muxer: MediaMuxer, track: Int, samples: List<Sample>) {
        val info = MediaCodec.BufferInfo()
        for (s in samples) {
            info.set(0, s.data.size, s.ptsUs, s.flags)
            muxer.writeSampleData(track, ByteBuffer.wrap(s.data), info)
        }
    }

    private fun encodeVideo(w: Int, h: Int, frames: Int, fps: Int, paint: (Int, RgbImage) -> Unit): Pair<MediaFormat, List<Sample>> {
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, w, h).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, 4_000_000)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
        var next = 0
        return try {
            drainAll(codec) { idx ->
                if (next < frames) {
                    val img = RgbImage(w, h).also { paint(next, it) }
                    val p = codec.getInputImage(idx)!!.planes
                    rgbToYuv420(
                        img,
                        Plane(p[0].buffer, p[0].rowStride, p[0].pixelStride),
                        Plane(p[1].buffer, p[1].rowStride, p[1].pixelStride),
                        Plane(p[2].buffer, p[2].rowStride, p[2].pixelStride),
                        YuvMatrix.BT601,
                    )
                    codec.queueInputBuffer(idx, 0, w * h * 3 / 2, next * 1_000_000L / fps, 0)
                    next++
                    false
                } else {
                    codec.queueInputBuffer(idx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    true
                }
            }
        } finally {
            codec.stop()
            codec.release()
        }
    }

    private fun encodeAudio(durationUs: Long): Pair<MediaFormat, List<Sample>> {
        val rate = 44_100
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
        val format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, rate, 1).apply {
            setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            setInteger(MediaFormat.KEY_BIT_RATE, 64_000)
        }
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
        val total = (durationUs * rate / 1_000_000L).toInt()
        var written = 0
        return try {
            drainAll(codec) { idx ->
                val buf = codec.getInputBuffer(idx)!!.order(ByteOrder.nativeOrder())
                val n = minOf(buf.remaining() / 2, total - written, 1024)
                if (n <= 0) {
                    codec.queueInputBuffer(idx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    true
                } else {
                    for (i in 0 until n) {
                        buf.putShort((sin(2 * PI * 440 * (written + i) / rate) * 8000).toInt().toShort())
                    }
                    codec.queueInputBuffer(idx, 0, n * 2, written * 1_000_000L / rate, 0)
                    written += n
                    false
                }
            }
        } finally {
            codec.stop()
            codec.release()
        }
    }

    /** Feed via [fill] (true once it queued end of stream) and collect output until the end. */
    private fun drainAll(codec: MediaCodec, fill: (Int) -> Boolean): Pair<MediaFormat, List<Sample>> {
        val out = ArrayList<Sample>()
        var format: MediaFormat? = null
        var inputDone = false
        val info = MediaCodec.BufferInfo()
        while (true) {
            if (!inputDone) {
                val idx = codec.dequeueInputBuffer(10_000)
                if (idx >= 0) inputDone = fill(idx)
            }
            val idx = codec.dequeueOutputBuffer(info, 10_000)
            if (idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                format = codec.outputFormat
            } else if (idx >= 0) {
                if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0 && info.size > 0) {
                    val b = codec.getOutputBuffer(idx)!!
                    val bytes = ByteArray(info.size)
                    (b as Buffer).position(info.offset) // Buffer: see OnnxInitializers
                    b.get(bytes)
                    out.add(Sample(bytes, info.presentationTimeUs, info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME))
                }
                codec.releaseOutputBuffer(idx, false)
                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
            }
        }
        return checkNotNull(format) { "Encoder never reported its format" } to out
    }
}
