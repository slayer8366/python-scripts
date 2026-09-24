package io.github.slayer8366.faceswap.core

import java.io.File
import java.io.RandomAccessFile
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Reads the last float initializer of an ONNX graph without loading the
 * model. insightface's INSwapper takes its embedding projection ("emap")
 * from `graph.initializer[-1]`; ONNX Runtime doesn't expose initializers,
 * so this walks the protobuf directly and skips everything else.
 */
object OnnxInitializers {
    class Tensor(val name: String, val dims: LongArray, val data: FloatArray)

    private const val MODEL_GRAPH = 7
    private const val GRAPH_INITIALIZER = 5
    private const val TENSOR_DIMS = 1
    private const val TENSOR_DATA_TYPE = 2
    private const val TENSOR_FLOAT_DATA = 4
    private const val TENSOR_NAME = 8
    private const val TENSOR_RAW_DATA = 9
    private const val TENSOR_DATA_LOCATION = 14
    private const val FLOAT = 1

    fun lastInitializer(model: File): Tensor =
        RandomAccessFile(model, "r").use { raf ->
            val buf = raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
            val graph = findLast(buf, 0, buf.limit(), MODEL_GRAPH) ?: error("No graph in $model")
            val init = findLast(buf, graph.first, graph.second, GRAPH_INITIALIZER)
                ?: error("Graph in $model has no initializers")
            parseTensor(buf, init.first, init.second)
        }

    /** Returns [start, end) of the last length-delimited field [field] in [start, end). */
    private fun findLast(buf: ByteBuffer, start: Int, end: Int, field: Int): Pair<Int, Int>? {
        var pos = start
        var found: Pair<Int, Int>? = null
        while (pos < end) {
            val (tag, p1) = varint(buf, pos)
            val num = (tag ushr 3).toInt(); val wire = (tag and 7).toInt()
            if (num == field && wire == 2) {
                val (len, p2) = varint(buf, p1)
                found = p2 to (p2 + len.toInt())
                pos = p2 + len.toInt()
            } else {
                pos = skip(buf, p1, wire)
            }
        }
        return found
    }

    private fun parseTensor(buf: ByteBuffer, start: Int, end: Int): Tensor {
        val dims = ArrayList<Long>()
        var name = ""
        var dataType = FLOAT
        var floats: FloatArray? = null
        val packedFloats = ArrayList<Float>()
        var pos = start
        while (pos < end) {
            val (tag, p1) = varint(buf, pos)
            val num = (tag ushr 3).toInt(); val wire = (tag and 7).toInt()
            pos = when {
                num == TENSOR_DIMS && wire == 0 -> varint(buf, p1).also { dims.add(it.first) }.second
                num == TENSOR_DIMS && wire == 2 -> {
                    val (len, p2) = varint(buf, p1)
                    var p = p2
                    while (p < p2 + len) p = varint(buf, p).also { dims.add(it.first) }.second
                    p
                }
                num == TENSOR_DATA_TYPE && wire == 0 -> varint(buf, p1).also { dataType = it.first.toInt() }.second
                num == TENSOR_NAME && wire == 2 -> {
                    val (len, p2) = varint(buf, p1)
                    val bytes = ByteArray(len.toInt())
                    val view = buf.duplicate()
                    // Through Buffer: ByteBuffer.position(int) doesn't exist on older Android.
                    (view as Buffer).position(p2)
                    view.get(bytes)
                    name = String(bytes, Charsets.UTF_8)
                    p2 + len.toInt()
                }
                (num == TENSOR_FLOAT_DATA || num == TENSOR_RAW_DATA) && wire == 2 -> {
                    val (len, p2) = varint(buf, p1)
                    floats = readFloats(buf, p2, len.toInt())
                    p2 + len.toInt()
                }
                num == TENSOR_FLOAT_DATA && wire == 5 -> {
                    packedFloats.add(readFloats(buf, p1, 4)[0]); p1 + 4
                }
                num == TENSOR_DATA_LOCATION && wire == 0 -> {
                    val (loc, p2) = varint(buf, p1)
                    require(loc == 0L) { "Tensor data is stored externally; not supported" }
                    p2
                }
                else -> skip(buf, p1, wire)
            }
        }
        require(dataType == FLOAT) { "Initializer $name has data type $dataType, expected float" }
        val data = floats ?: packedFloats.toFloatArray()
        val expected = dims.fold(1L) { a, b -> a * b }
        require(data.size.toLong() == expected) { "Initializer $name has ${data.size} values, dims say $expected" }
        return Tensor(name, dims.toLongArray(), data)
    }

    private fun readFloats(buf: ByteBuffer, pos: Int, len: Int): FloatArray {
        val view = buf.duplicate()
        (view as Buffer).position(pos)
        (view as Buffer).limit(pos + len)
        return FloatArray(len / 4).also { view.slice().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
    }

    private fun varint(buf: ByteBuffer, start: Int): Pair<Long, Int> {
        var result = 0L
        var shift = 0
        var pos = start
        while (true) {
            val b = buf.get(pos++).toInt()
            result = result or ((b and 0x7F).toLong() shl shift)
            if (b and 0x80 == 0) return result to pos
            shift += 7
            require(shift < 64) { "Malformed varint at $start" }
        }
    }

    private fun skip(buf: ByteBuffer, pos: Int, wire: Int): Int = when (wire) {
        0 -> varint(buf, pos).second
        1 -> pos + 8
        2 -> varint(buf, pos).let { (len, p) -> p + len.toInt() }
        5 -> pos + 4
        else -> error("Unsupported protobuf wire type $wire")
    }
}
