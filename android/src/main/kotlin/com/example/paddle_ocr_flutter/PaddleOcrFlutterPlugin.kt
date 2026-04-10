package com.example.paddle_ocr_flutter

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import java.io.File
import java.io.FileOutputStream
import java.util.Vector
import java.util.concurrent.atomic.AtomicBoolean

class PaddleOcrFlutterPlugin : FlutterPlugin, MethodCallHandler {

    private lateinit var channel: MethodChannel
    private lateinit var context: Context

    private var nativePointer: Long = 0
    private var isModelLoaded = false
    private val wordLabels = Vector<String>()

    companion object {
        private const val TAG = "PaddleOCR"
        private val isSOLoaded = AtomicBoolean(false)

        private fun loadNativeLibrary() {
            if (!isSOLoaded.get() && isSOLoaded.compareAndSet(false, true)) {
                try {
                    System.loadLibrary("paddle_ocr_jni")
                    Log.i(TAG, "Native library loaded")
                } catch (e: Throwable) {
                    isSOLoaded.set(false)
                    throw RuntimeException("Load libpaddle_ocr_jni.so failed", e)
                }
            }
        }
    }

    // JNI — signature must match native.cpp
    private external fun nativeInit(
        detModelPath: String,
        recModelPath: String,
        clsModelPath: String,
        threadNum: Int
    ): Long

    private external fun nativeForward(
        pointer: Long,
        bitmap: Bitmap,
        maxSizeLen: Int,
        runDet: Int,
        runCls: Int,
        runRec: Int
    ): FloatArray

    private external fun nativeRelease(pointer: Long)

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(binding.binaryMessenger, "paddle_ocr_flutter")
        channel.setMethodCallHandler(this)
        context = binding.applicationContext
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "init" -> handleInit(call, result)
            "recognize" -> handleRecognize(call, result)
            "release" -> handleRelease(result)
            else -> result.notImplemented()
        }
    }

    private fun handleInit(call: MethodCall, result: Result) {
        try {
            loadNativeLibrary()

            val threadNum = call.argument<Int>("threadNum") ?: 4
            val modelDir = call.argument<String>("modelDir") ?: "models"
            val labelPath = call.argument<String>("labelPath") ?: "labels/ppocr_keys_v1.txt"

            val cacheDir = File(context.cacheDir, "paddle_ocr")
            if (!cacheDir.exists()) cacheDir.mkdirs()

            // Flutter plugin assets are at: flutter_assets/packages/<plugin>/assets/...
            val assetPrefix = "flutter_assets/packages/paddle_ocr_flutter/assets"
            val detPath = copyAssetToCache("$assetPrefix/$modelDir/det_v5.onnx", cacheDir)
            val recPath = copyAssetToCache("$assetPrefix/$modelDir/rec_v5.onnx", cacheDir)
            val clsPath = copyAssetToCache("$assetPrefix/$modelDir/cls_v2.onnx", cacheDir)

            loadLabels("$assetPrefix/$labelPath")

            nativePointer = nativeInit(detPath, recPath, clsPath, threadNum)
            isModelLoaded = nativePointer != 0L

            if (isModelLoaded) {
                Log.i(TAG, "Model loaded, pointer=$nativePointer, labels=${wordLabels.size}")
                result.success(mapOf("success" to true, "labelCount" to wordLabels.size))
            } else {
                result.error("INIT_FAILED", "Native init returned null pointer", null)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Init failed", e)
            result.error("INIT_FAILED", e.message, null)
        }
    }

    private fun handleRecognize(call: MethodCall, result: Result) {
        if (!isModelLoaded || nativePointer == 0L) {
            result.error("NOT_INITIALIZED", "Model not loaded, call init first", null)
            return
        }

        try {
            val imagePath = call.argument<String>("imagePath")
                ?: return result.error("INVALID_ARGS", "imagePath required", null)
            val maxSizeLen = call.argument<Int>("maxSizeLen") ?: 960

            val bitmap = BitmapFactory.decodeFile(imagePath)
                ?: return result.error("DECODE_FAILED", "Cannot decode image: $imagePath", null)

            val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            bitmap.recycle()

            val rawResults = nativeForward(nativePointer, argbBitmap, maxSizeLen, 1, 1, 1)
            argbBitmap.recycle()

            val ocrResults = parseResults(rawResults)
            result.success(ocrResults)
        } catch (e: Exception) {
            Log.e(TAG, "Recognize failed", e)
            result.error("RECOGNIZE_FAILED", e.message, null)
        }
    }

    private fun handleRelease(result: Result) {
        if (nativePointer != 0L) {
            nativeRelease(nativePointer)
            nativePointer = 0
            isModelLoaded = false
        }
        result.success(true)
    }

    private fun parseResults(raw: FloatArray): List<Map<String, Any>> {
        val results = mutableListOf<Map<String, Any>>()
        var pos = 0

        while (pos < raw.size) {
            val pointNum = Math.round(raw[pos]).toInt()
            val wordNum = Math.round(raw[pos + 1]).toInt()
            val score = raw[pos + 2]
            pos += 3

            val points = mutableListOf<Map<String, Int>>()
            for (i in 0 until pointNum) {
                points.add(mapOf("x" to Math.round(raw[pos]).toInt(), "y" to Math.round(raw[pos + 1]).toInt()))
                pos += 2
            }

            val sb = StringBuilder()
            for (i in 0 until wordNum) {
                val idx = Math.round(raw[pos]).toInt()
                if (idx >= 0 && idx < wordLabels.size) {
                    sb.append(wordLabels[idx])
                }
                pos++
            }

            val clsLabel = Math.round(raw[pos]).toInt()
            val clsScore = raw[pos + 1]
            pos += 2

            results.add(mapOf(
                "text" to sb.toString(),
                "confidence" to score,
                "points" to points,
                "clsLabel" to clsLabel,
                "clsScore" to clsScore
            ))
        }
        return results
    }

    private fun loadLabels(labelPath: String) {
        wordLabels.clear()
        wordLabels.add("blank") // CTC blank at index 0
        try {
            val input = context.assets.open(labelPath)
            val content = input.bufferedReader().readText()
            input.close()
            content.split("\n").forEach { line ->
                if (line.isNotEmpty()) wordLabels.add(line)
            }
            wordLabels.add(" ") // space at end
            Log.i(TAG, "Labels loaded: ${wordLabels.size}")
        } catch (e: Exception) {
            Log.e(TAG, "Load labels failed", e)
        }
    }

    private fun copyAssetToCache(assetPath: String, cacheDir: File): String {
        val outFile = File(cacheDir, assetPath.substringAfterLast("/"))
        if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath

        context.assets.open(assetPath).use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
        return outFile.absolutePath
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        if (nativePointer != 0L) {
            nativeRelease(nativePointer)
            nativePointer = 0
        }
    }
}
