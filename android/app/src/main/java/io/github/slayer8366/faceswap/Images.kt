package io.github.slayer8366.faceswap

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageDecoder
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import io.github.slayer8366.faceswap.core.RgbImage
import kotlin.math.max
import kotlin.math.roundToInt

const val DEFAULT_LABEL = "AI-GENERATED: face swapped"

/** Decode a picked photo, honouring EXIF rotation, capped at [maxSide] pixels. */
fun decodeImage(context: Context, uri: Uri, maxSide: Int = 1920): Bitmap {
    val source = ImageDecoder.createSource(context.contentResolver, uri)
    val bitmap = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        val longest = max(info.size.width, info.size.height)
        if (longest > maxSide) {
            val s = maxSide.toFloat() / longest
            decoder.setTargetSize((info.size.width * s).roundToInt(), (info.size.height * s).roundToInt())
        }
    }
    return if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap else bitmap.copy(Bitmap.Config.ARGB_8888, false)
}

fun Bitmap.toRgbImage(): RgbImage {
    val px = IntArray(width * height)
    getPixels(px, 0, width, 0, 0, width, height)
    val img = RgbImage(width, height)
    for (i in px.indices) {
        val p = px[i]
        img.data[i * 3] = (p shr 16).toByte()
        img.data[i * 3 + 1] = (p shr 8).toByte()
        img.data[i * 3 + 2] = p.toByte()
    }
    return img
}

fun RgbImage.toBitmap(): Bitmap {
    val px = IntArray(width * height) { i ->
        val r = data[i * 3].toInt() and 0xFF
        val g = data[i * 3 + 1].toInt() and 0xFF
        val b = data[i * 3 + 2].toInt() and 0xFF
        Color.rgb(r, g, b)
    }
    return Bitmap.createBitmap(px, width, height, Bitmap.Config.ARGB_8888)
}

/** The visible disclosure label, pre-rendered once per output size. */
class LabelOverlay(text: String, frameWidth: Int, frameHeight: Int) {
    val pixels: IntArray
    val width: Int
    val height: Int
    val x: Int
    val y: Int

    init {
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = max(14f, frameHeight / 32f)
            isFakeBoldText = true
        }
        val pad = (paint.textSize * 0.4f).roundToInt()
        val fm = paint.fontMetrics
        width = minOf(frameWidth, paint.measureText(text).roundToInt() + 2 * pad)
        height = minOf(frameHeight, (fm.descent - fm.ascent).roundToInt() + 2 * pad)
        x = pad
        y = max(0, frameHeight - height - pad)
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        Canvas(bmp).apply {
            drawRoundRect(RectF(0f, 0f, width.toFloat(), height.toFloat()), pad / 2f, pad / 2f,
                Paint().apply { color = Color.argb(166, 0, 0, 0) })
            drawText(text, pad.toFloat(), pad - fm.ascent, paint)
        }
        pixels = IntArray(width * height).also { bmp.getPixels(it, 0, width, 0, 0, width, height) }
        bmp.recycle()
    }

    fun drawOn(img: RgbImage) = img.blendArgb(pixels, width, height, x, y)
}
