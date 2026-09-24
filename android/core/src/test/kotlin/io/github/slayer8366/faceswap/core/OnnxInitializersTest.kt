package io.github.slayer8366.faceswap.core

import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class OnnxInitializersTest {
    private fun ByteArrayOutputStream.varint(v: Long) {
        var x = v
        while (x >= 0x80) { write(((x and 0x7F) or 0x80).toInt()); x = x ushr 7 }
        write(x.toInt())
    }

    private fun field(num: Int, wire: Int, body: ByteArrayOutputStream.() -> Unit) =
        ByteArrayOutputStream().apply { varint(((num shl 3) or wire).toLong()); body() }.toByteArray()

    private fun bytesField(num: Int, bytes: ByteArray) = field(num, 2) { varint(bytes.size.toLong()); write(bytes) }

    private fun floatsLe(vararg f: Float) =
        ByteBuffer.allocate(f.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { f.forEach { putFloat(it) } }.array()

    private fun tensor(name: String, dims: List<Long>, data: FloatArray, raw: Boolean, packedDims: Boolean): ByteArray {
        val out = ByteArrayOutputStream()
        if (packedDims) {
            val d = ByteArrayOutputStream().apply { dims.forEach { varint(it) } }.toByteArray()
            out.write(bytesField(1, d))
        } else {
            dims.forEach { dim -> out.write(field(1, 0) { varint(dim) }) }
        }
        out.write(field(2, 0) { varint(1) })
        out.write(bytesField(8, name.toByteArray()))
        out.write(bytesField(if (raw) 9 else 4, floatsLe(*data)))
        return out.toByteArray()
    }

    private fun model(vararg tensors: ByteArray): File {
        val graph = ByteArrayOutputStream()
        graph.write(bytesField(1, byteArrayOf(1, 2, 3))) // a "node" to skip
        tensors.forEach { graph.write(bytesField(5, it)) }
        graph.write(bytesField(2, "g".toByteArray()))
        val m = ByteArrayOutputStream()
        m.write(field(1, 0) { varint(8) }) // ir_version
        m.write(bytesField(2, "test".toByteArray()))
        m.write(bytesField(7, graph.toByteArray()))
        m.write(field(5, 0) { varint(300) })
        return File.createTempFile("model", ".onnx").apply { writeBytes(m.toByteArray()); deleteOnExit() }
    }

    @Test
    fun readsLastInitializerInEveryEncoding() {
        for (raw in listOf(true, false)) for (packed in listOf(true, false)) {
            val f = model(
                tensor("first", listOf(2), floatArrayOf(9f, 9f), raw, packed),
                tensor("emap", listOf(2, 3), floatArrayOf(1f, 2f, 3f, 4f, 5f, -6.5f), raw, packed),
            )
            val t = OnnxInitializers.lastInitializer(f)
            assertEquals("emap", t.name)
            assertContentEquals(longArrayOf(2, 3), t.dims)
            assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f, 5f, -6.5f), t.data)
        }
    }

    @Test
    fun rejectsSizeMismatch() {
        val f = model(tensor("bad", listOf(4), floatArrayOf(1f), raw = true, packedDims = true))
        assertFailsWith<IllegalArgumentException> { OnnxInitializers.lastInitializer(f) }
    }
}
