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
/**
 * Which swap model to use. FAST is an int8 build of inswapper_128: about 2.7x
 * faster on CPU, with an identity-similarity drop of 0.005 and a mean pixel
 * difference of 2.4/255 against FULL on held-out faces (desktop measurements).
 */
enum class SwapModel(val label: String, val downloadMb: Int) {
    FAST("Fast (int8)", 466),
    FULL("Full precision", 843),
}

class ModelStore(context: Context) {
    private val dir = File(context.noBackupFilesDir, "models")

    private class Asset(val file: String, val sha256: String)

    private val detector = Asset("det_10g.onnx", "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91")
    private val recognizer = Asset("w600k_r50.onnx", "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43")
    private val swapper = Asset("inswapper_128.onnx", "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af")
    // Built from the official model by android/tools/quantize_swapper.py in the
    // swap-model-int8 workflow; the build is reproducible byte for byte.
    private val swapperInt8 = Asset("inswapper_128_int8.onnx", "cfbfd8518e79a1f672550963be173936b6ee58130ee757aec08094335d08cd65")
    private val emapAsset = Asset("inswapper_emap.bin", "4e823c24cc60c3796fcd5bc68ab3743e7357aa9c97af9f9ebb5f4234d9bd5c1e")
    private val emapCache = File(dir, emapAsset.file)

    private fun has(a: Asset) = File(dir, a.file).isFile

    fun isReady(variant: SwapModel): Boolean = has(detector) && has(recognizer) && when (variant) {
        SwapModel.FAST -> has(swapperInt8) && has(emapAsset)
        SwapModel.FULL -> has(swapper)
    }

    /** Download whatever [variant] still needs. [progress] gets (bytes done, bytes total). */
    fun download(variant: SwapModel, progress: (Long, Long) -> Unit, cancelled: () -> Boolean) {
        dir.mkdirs()
        val needPack = !has(detector) || !has(recognizer)
        val singles = when (variant) {
            SwapModel.FAST -> listOf(swapperInt8 to (FAST_URL to INT8_BYTES), emapAsset to (FAST_URL to EMAP_BYTES))
            SwapModel.FULL -> listOf(swapper to (RELEASE_URL to SWAPPER_BYTES))
        }.filterNot { has(it.first) }
        val total = singles.sumOf { it.second.second } + if (needPack) PACK_BYTES else 0L
        var done = 0L
        val report = { n: Long -> done += n; progress(done, total) }

        for ((asset, source) in singles) {
            open(source.first + asset.file).use { input -> save(input, asset, report, cancelled) }
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
            check(has(detector) && has(recognizer)) { "buffalo_l.zip did not contain the expected models" }
        }
    }

    /** Model paths plus the swapper's embedding projection. */
    fun load(variant: SwapModel): ModelFiles {
        check(isReady(variant)) { "Models are not downloaded" }
        val swapFile = File(dir, if (variant == SwapModel.FAST) swapperInt8.file else swapper.file)
        // The fast variant downloads the projection (quantization drops it from
        // the graph); the full model carries it, extracted once and cached in
        // the same big-endian format.
        if (emapCache.length() != 512L * 512 * 4) {
            val data = OnnxInitializers.lastInitializer(File(dir, swapper.file)).data
            DataOutputStream(emapCache.outputStream().buffered()).use { s -> data.forEach(s::writeFloat) }
        }
        val emap = DataInputStream(emapCache.inputStream().buffered()).use { s -> FloatArray(512 * 512) { s.readFloat() } }
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
        const val FAST_URL = "https://github.com/slayer8366/python-scripts/releases/download/swap-model-int8-v1/"
        const val INT8_BYTES = 176_530_283L
        const val EMAP_BYTES = 1_048_576L
        const val LICENSE_NOTICE =
            "Models: InsightFace buffalo_l and inswapper_128 (and the int8 build of it), licensed for " +
                "non-commercial research use only."
    }
}
