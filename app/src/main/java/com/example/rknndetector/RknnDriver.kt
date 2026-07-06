package com.example.rknndetector

import android.content.res.AssetManager

class RknnDriver {
    // 加载 Assets 中的模型（传入字节数组）
    external fun initModel(modelBuf: ByteArray, modelSize: Int): Boolean

    // 获取模型输入尺寸 [Width, Height]
    external fun getInputShape(): IntArray?

    // 执行推理
    external fun runInference(rgbPixels: ByteArray, width: Int, height: Int): FloatArray?

    // 释放 NPU 资源
    external fun releaseModel()

    // 辅助函数：从 Assets 读取模型文件
    fun loadModelFromAssets(assetManager: AssetManager, fileName: String): Boolean {
        return try {
            val inputStream = assetManager.open(fileName)
            val length = inputStream.available()
            val buffer = ByteArray(length)
            inputStream.read(buffer)
            inputStream.close()
            initModel(buffer, length)
        } catch (e: Exception) {
            android.util.Log.e("RknnDriver", "加载模型失败: ${e.message}")
            false
        }
    }

    companion object {
        init {
            System.loadLibrary("rknndetector")
        }
    }
}
