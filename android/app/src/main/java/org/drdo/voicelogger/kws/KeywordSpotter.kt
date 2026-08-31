package org.drdo.voicelogger.kws

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import org.drdo.voicelogger.anc.FFT512

/**
 * On-Device Keyword Spotting Engine.
 * NOTE: Disabled by default (ENABLE_KEYWORD_SPOTTING = false) until a fully trained custom wake-word
 * model (e.g., Picovoice Porcupine or trained TFLite/ONNX classifier) is supplied.
 */
class KeywordSpotter(
    private val context: Context,
    var confidenceThreshold: Float = 0.85f
) {
    companion object {
        /**
         * Feature flag for Keyword Spotting. Set to false until trained weights are provided.
         */
        const val ENABLE_KEYWORD_SPOTTING = false
    }

    interface Callback {
        fun onCodeWordDetected(codeWord: String, confidence: Float)
    }

    val targetCodeWords = listOf("RED_TACTICAL", "PHOENIX", "MAYDAY")
    private var callback: Callback? = null

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var isInitialized = false

    init {
        if (ENABLE_KEYWORD_SPOTTING) {
            initEngine()
        } else {
            android.util.Log.i("KWS", "Keyword Spotting feature flag is OFF (requires trained wake-word model).")
        }
    }

    private fun initEngine() {
        try {
            env = OrtEnvironment.getEnvironment()
            val modelFile = getModelFile()
            if (modelFile.exists()) {
                session = env?.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
                isInitialized = true
                android.util.Log.i("KWS", "ONNX Keyword Spotting engine initialized successfully.")
            } else {
                android.util.Log.e("KWS", "KWS Engine Initialization Failed: Model asset kws_model.onnx not found!")
            }
        } catch (e: Throwable) {
            // Throwable for the same reason as OnnxAncProcessor: a native loader Error here
            // would escape the constructor and kill the capture service.
            isInitialized = false
            android.util.Log.e("KWS", "KWS engine init failed: " + e.message, e)
        }
    }

    private fun getModelFile(): File {
        val file = File(context.filesDir, "kws_model.onnx")
        if (!file.exists()) {
            try {
                context.assets.open("kws_model.onnx").use { input ->
                    FileOutputStream(file).use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("KWS", "Could not copy kws_model.onnx from assets: ${e.message}")
            }
        }
        return file
    }

    fun setCallback(cb: Callback) {
        this.callback = cb
    }

    /**
     * Processes PCM16 audio frame through ONNX KWS model if feature flag is ON.
     */
    fun processFrame(pcm16Samples: ShortArray): Boolean {
        if (!ENABLE_KEYWORD_SPOTTING || !isInitialized || session == null || env == null) {
            return false
        }

        if (pcm16Samples.size < 512) {
            return false
        }

        try {
            val realIn = FloatArray(512)
            val imagIn = FloatArray(512)
            for (i in 0 until 512) {
                realIn[i] = pcm16Samples[i] / 32768.0f
            }

            val realOut = FloatArray(512)
            val imagOut = FloatArray(512)
            FFT512.fft(realIn, imagIn, realOut, imagOut)

            val features = FloatArray(257)
            for (i in 0 until 257) {
                features[i] = kotlin.math.sqrt(realOut[i] * realOut[i] + imagOut[i] * imagOut[i])
            }

            val shape = longArrayOf(1, 257)
            val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(features), shape)
            val results = session?.run(mapOf("input_features" to tensor))
            val outputTensor = results?.get(0)?.value as? Array<FloatArray>

            tensor.close()
            results?.close()

            if (outputTensor != null && outputTensor.isNotEmpty() && outputTensor[0].size >= 5) {
                val probs = outputTensor[0]

                var maxIdx = 0
                var maxProb = 0.0f
                for (i in probs.indices) {
                    if (probs[i] > maxProb) {
                        maxProb = probs[i]
                        maxIdx = i
                    }
                }

                if (maxIdx >= 2 && maxProb >= confidenceThreshold) {
                    val detectedWord = targetCodeWords[maxIdx - 2]
                    android.util.Log.w("KWS", "Duress code word '$detectedWord' detected by neural ONNX engine (Real Score: $maxProb)")
                    callback?.onCodeWordDetected(detectedWord, maxProb)
                    return true
                }
            }
        } catch (e: Exception) {
            android.util.Log.e("KWS", "KWS Inference exception: ${e.message}")
        }

        return false
    }
}
