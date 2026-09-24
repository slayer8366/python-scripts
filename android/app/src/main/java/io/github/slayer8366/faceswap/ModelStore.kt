package io.github.slayer8366.faceswap

import android.content.Context
import io.github.slayer8366.faceswap.core.ModelFiles
import io.github.slayer8366.faceswap.core.OnnxInitializers
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/**
 * Downloads and locates the InsightFace models. Files come from the official
 * `deepinsight/insightface` model-zoo release and are checked against
 * SHA-256 hashes before use. The weights are licensed for non-commercial
 * research only.
 */
class ModelStore(context: Context) {
    private val dir = File(context.noBackupFilesDir, "models")

    private class Asset(val file: String, val sha256: String)

    private val detector = Asset("det_10g.onnx", "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91")
    private val recognizer = Asset("w600k_r50.onnx", "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43")
    private val swapper = Asset("inswapper_128.onnx", "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af")
    private val emapCache = File(dir, "inswapper_emap.bin")

    val isReady: Boolean
        get() = listOf(detector, recognizer, swapper).all { File(dir, it.file).isFile }

    /** Download whatever is missing. [progress] gets (bytes done, bytes total). */
    fun download(progress: (Long, Long) -> Unit, cancelled: () -> Boolean) {
        dir.mkdirs()
        val needSwapper = !File(dir, swapper.file).isFile
        val needPack = !File(dir, detector.file).isFile || !File(dir, recognizer.file).isFile
        val total = (if (needSwapper) SWAPPER_BYTES else 0L) + (if (needPack) PACK_BYTES else 0L)
        var done = 0L
        val report = { n: Long -> done += n; progress(done, total) }

        if (needSwapper) {
            open(RELEASE_URL + swapper.file).use { input -> save(input, swapper, report, cancelled) }
        }
        if (needPack) {
            // Stream the zip and keep only the two files we need, so the
            // 288 MB archive never has to sit on disk.
            open(RELEASE_URL + "buffalo_l.zip").use { raw ->
                val counting = CountingStream(raw, report)
                ZipInputStream(counting).use { zip ->
                    val wanted = listOf(detector, recognizer).associateBy { it.file }
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        val asset = wanted[entry.name.substringAfterLast('/')] ?: continue
                        save(zip, asset, {}, cancelled, closeInput = false)
                    }
                }
            }
            check(File(dir, detector.file).isFile && File(dir, recognizer.file).isFile) {
                "buffalo_l.zip did not contain the expected models"
            }
        }
    }

    /** Model paths plus the swapper projection, extracted once and cached. */
    fun load(): ModelFiles {
        check(isReady) { "Models are not downloaded" }
        val swapFile = File(dir, swapper.file)
        val emap = if (emapCache.length() == 512L * 512 * 4) {
            DataInputStream(emapCache.inputStream().buffered()).use { s -> FloatArray(512 * 512) { s.readFloat() } }
        } else {
            OnnxInitializers.lastInitializer(swapFile).data.also { data ->
                DataOutputStream(emapCache.outputStream().buffered()).use { s -> data.forEach(s::writeFloat) }
            }
        }
        return ModelFiles(File(dir, detector.file), File(dir, recognizer.file), swapFile, emap)
    }

    private fun open(url: String): InputStream {
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.connectTimeout = 30_000
        conn.readTimeout = 60_000
        if (conn.responseCode != HttpURLConnection.HTTP_OK) {
            conn.disconnect()
            throw java.io.IOException("Download failed: HTTP ${conn.responseCode} for $url")
        }
        return conn.inputStream
    }

    private fun save(
        input: InputStream,
        asset: Asset,
        report: (Long) -> Unit,
        cancelled: () -> Boolean,
        closeInput: Boolean = true,
    ) {
        val part = File(dir, asset.file + ".part")
        val digest = MessageDigest.getInstance("SHA-256")
        try {
            part.outputStream().use { out ->
                val buf = ByteArray(1 shl 16)
                while (true) {
                    if (cancelled()) throw InterruptedException("Download cancelled")
                    val n = input.read(buf)
                    if (n < 0) break
                    digest.update(buf, 0, n)
                    out.write(buf, 0, n)
                    report(n.toLong())
                }
            }
            val hex = digest.digest().joinToString("") { "%02x".format(it) }
            if (hex != asset.sha256) throw SecurityException("Checksum mismatch for ${asset.file}; file discarded")
            check(part.renameTo(File(dir, asset.file))) { "Could not move ${asset.file} into place" }
        } finally {
            part.delete()
            if (closeInput) input.close()
        }
    }

    private class CountingStream(private val inner: InputStream, private val report: (Long) -> Unit) : InputStream() {
        override fun read(): Int = inner.read().also { if (it >= 0) report(1) }
        override fun read(b: ByteArray, off: Int, len: Int): Int = inner.read(b, off, len).also { if (it > 0) report(it.toLong()) }
        override fun close() = inner.close()
    }

    companion object {
        const val RELEASE_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/"
        const val SWAPPER_BYTES = 554_253_681L
        const val PACK_BYTES = 288_621_354L
        const val LICENSE_NOTICE =
            "Models: InsightFace buffalo_l and inswapper_128, licensed for non-commercial research use only."
    }
}
