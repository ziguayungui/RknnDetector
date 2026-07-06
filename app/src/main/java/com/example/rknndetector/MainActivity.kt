package com.example.rknndetector

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.rknndetector.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val rknnDriver = RknnDriver()

    private var modelWidth = 640
    private var modelHeight = 640
    private var isModelReady = false

    private val labels = arrayOf("head", "mask")
    private val labelColors = intArrayOf(Color.BLUE, Color.GREEN)

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK && result.data != null) {
            result.data!!.data?.let { uri -> processImage(uri) }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val modelFileName = "yolov5_mask_rk3576.rknn"
        val loadSuccess = rknnDriver.loadModelFromAssets(assets, modelFileName)

        if (loadSuccess) {
            val shape = rknnDriver.getInputShape()
            if (shape != null && shape.size >= 2) {
                modelWidth = shape[0]
                modelHeight = shape[1]
                isModelReady = true
                binding.tvStatus.text = "模型就绪，输入尺寸: ${modelWidth}x${modelHeight}"
                Toast.makeText(this, "模型加载成功！", Toast.LENGTH_SHORT).show()
            } else {
                binding.tvStatus.text = "模型加载失败: 无法获取尺寸"
            }
        } else {
            binding.tvStatus.text = "模型加载失败，请检查 assets 中的文件"
            Toast.makeText(this, "rknn_init 失败，请查看 logcat", Toast.LENGTH_LONG).show()
        }

        binding.btnSelectImage.setOnClickListener {
            if (!isModelReady) {
                Toast.makeText(this, "模型未就绪", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
                type = "image/*"
                addCategory(Intent.CATEGORY_OPENABLE)
            }
            pickImageLauncher.launch(intent)
        }
    }

    private fun processImage(uri: Uri) {
        android.util.Log.d("RKNN_NPU", "processImage: modelWidth=$modelWidth, modelHeight=$modelHeight")
        
        val inputStream = contentResolver.openInputStream(uri) ?: return
        val options = BitmapFactory.Options().apply {
            inScaled = false
            inSampleSize = 1
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val srcBitmap = BitmapFactory.decodeStream(inputStream, null, options)
        inputStream.close()
        if (srcBitmap == null) {
            Toast.makeText(this, "图片解码失败", Toast.LENGTH_SHORT).show()
            return
        }
        android.util.Log.d("RKNN_NPU", "processImage: srcBitmap=${srcBitmap.width}x${srcBitmap.height}")

        val resizedBitmap = Bitmap.createScaledBitmap(srcBitmap, modelWidth, modelHeight, true)
        android.util.Log.d("RKNN_NPU", "processImage: resizedBitmap=${resizedBitmap.width}x${resizedBitmap.height}")
        
        val rgbBytes = bitmapToRGB(resizedBitmap)
        android.util.Log.d("RKNN_NPU", "processImage: rgbBytes size=${rgbBytes.size}, expected=${modelWidth * modelHeight * 3}")

        val startTime = System.currentTimeMillis()
        val finalBoxes = rknnDriver.runInference(rgbBytes, modelWidth, modelHeight)
        val costTime = System.currentTimeMillis() - startTime

        val boxCount = if (finalBoxes == null) 0 else finalBoxes.size / 6
        binding.tvStatus.text = "推理耗时: ${costTime}ms, 检出目标: $boxCount"

        if (finalBoxes != null) {
            val drawBitmap = drawBoxesOnBitmap(srcBitmap, finalBoxes)
            binding.imageView.setImageBitmap(drawBitmap)
        } else {
            Toast.makeText(this, "推理异常，请查看 logcat", Toast.LENGTH_SHORT).show()
        }
    }

    private fun bitmapToRGB(bitmap: Bitmap): ByteArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val rgb = ByteArray(bitmap.width * bitmap.height * 3)
        var idx = 0
        for (pixel in pixels) {
            rgb[idx++] = ((pixel shr 16) and 0xFF).toByte()
            rgb[idx++] = ((pixel shr 8) and 0xFF).toByte()
            rgb[idx++] = (pixel and 0xFF).toByte()
        }
        return rgb
    }

    private fun drawBoxesOnBitmap(bitmap: Bitmap, boxes: FloatArray): Bitmap {
        val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)
        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 4.0f
            textSize = 30.0f
            isAntiAlias = true
        }

        val textPaint = Paint().apply {
            textSize = 24.0f
            isAntiAlias = true
            style = Paint.Style.FILL
        }

        for (i in 0 until boxes.size / 6) {
            val b = i * 6
            val x1 = boxes[b] * bitmap.width
            val y1 = boxes[b + 1] * bitmap.height
            val x2 = boxes[b + 2] * bitmap.width
            val y2 = boxes[b + 3] * bitmap.height
            val score = boxes[b + 4]
            val classId = boxes[b + 5].toInt()

            val color = if (classId >= 0 && classId < labelColors.size) labelColors[classId] else Color.RED
            val label = if (classId >= 0 && classId < labels.size) labels[classId] else "unknown"

            paint.color = color
            canvas.drawRect(x1, y1, x2, y2, paint)

            val text = "$label ${(score * 100).toInt()}%"
            textPaint.color = color
            val textWidth = textPaint.measureText(text)
            val bgPaint = Paint().apply {
                this.color = color
                style = Paint.Style.FILL
            }
            canvas.drawRect(x1, y1 - 35, x1 + textWidth + 10, y1, bgPaint)
            textPaint.color = Color.WHITE
            canvas.drawText(text, x1 + 5, y1 - 10, textPaint)
        }
        return mutable
    }

    override fun onDestroy() {
        super.onDestroy()
        rknnDriver.releaseModel()
    }
}
