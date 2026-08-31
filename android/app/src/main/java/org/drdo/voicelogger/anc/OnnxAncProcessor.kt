package org.drdo.voicelogger.anc

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

/**
 * Result data class for On-Device Noise Cancellation.
 * STRICT SPEC REQUIREMENT: Never report an unmeasured or fake SNR.
 * levelReductionDb represents 10*log10(in_power / out_power) - a LEVEL change, NOT SNR.
 */
data class AncResult(
    val enhancedPcm: FloatArray,
    val levelReductionDb: Double?, // null if processing failed
    val processed: Boolean,
    val failureReason: String? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as AncResult
        return enhancedPcm.contentEquals(other.enhancedPcm) &&
                levelReductionDb == other.levelReductionDb &&
                processed == other.processed &&
                failureReason == other.failureReason
    }

    override fun hashCode(): Int {
        var result = enhancedPcm.contentHashCode()
        result = 31 * result + (levelReductionDb?.hashCode() ?: 0)
        result = 31 * result + processed.hashCode()
        result = 31 * result + (failureReason?.hashCode() ?: 0)
        return result
    }
}

/**
 * Real-Time Streaming ONNX Neural Noise Cancellation Processor.
 * Contract:
 * Inputs:
 *  - noisy_real (1, 257, T) float32
 *  - noisy_imag (1, 257, T) float32
 *  - h_state (2, 1, 128) float32
 * Outputs:
 *  - enh_real, enh_imag, mask_real, mask_imag, next_h_state
 */
class OnnxAncProcessor(private val context: Context) {
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private val stftEngine = StftEngine()

    private var isInitialized = false
    private var initFailureReason: String? = null

    // Streaming GRU hidden state (2, 1, 128 = 256 floats) maintained across chunks
    private var currentHState = FloatArray(2 * 1 * 128)

    init {
        initOnnxSession()
    }

    private fun initOnnxSession() {
        try {
            env = OrtEnvironment.getEnvironment()
            val modelFile = getModelFile()
            if (modelFile.exists()) {
                session = env?.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
                isInitialized = true
                android.util.Log.i("ANC", "ONNX Causal-CRN ANC session initialized successfully.")
            } else {
                isInitialized = false
                initFailureReason = "Model asset causal_anc_model.onnx not found in assets."
                android.util.Log.e("ANC", initFailureReason!!)
            }
        } catch (e: Throwable) {
            // Throwable, NOT Exception. Loading libonnxruntime.so can fail with
            // UnsatisfiedLinkError or NoClassDefFoundError, which are Errors and would sail
            // straight past a catch(Exception) - out of this constructor, out of the service's
            // onCreate, and into a START_STICKY restart loop that the user sees as
            // "DRDO Voice Logger keeps stopping" with no recording happening at all.
            // Noise cancellation is an enhancement; capture is the duty. Losing the first must
            // never cost the second.
            isInitialized = false
            initFailureReason = "ONNX session init failed: " + e.javaClass.simpleName +
                    ": " + (e.message ?: "no detail")
            android.util.Log.e("ANC", initFailureReason!!, e)
        }
    }

    private fun getModelFile(): File {
        val file = File(context.filesDir, "causal_anc_model.onnx")
        if (!file.exists()) {
            try {
                context.assets.open("causal_anc_model.onnx").use { input ->
                    FileOutputStream(file).use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("ANC", "Could not copy causal_anc_model.onnx from assets: ${e.message}")
            }
        }
        return file
    }

    /**
     * Resets the streaming recurrent GRU state to zeros.
     */
    fun resetState() {
        currentHState.fill(0.0f)
    }

    /**
     * Executes STFT -> 3 Named ONNX Inputs -> Streaming h_state -> iSTFT pipeline.
     */
    fun processAudio(inputPcm: FloatArray): AncResult {
        if (!isInitialized || session == null || env == null) {
            return AncResult(
                enhancedPcm = inputPcm.clone(),
                levelReductionDb = null,
                processed = false,
                failureReason = initFailureReason ?: "ONNX Processor uninitialized"
            )
        }

        if (inputPcm.isEmpty()) {
            return AncResult(inputPcm.clone(), null, false, "Input PCM is empty")
        }

        try {
            // 1. Compute 512-point Hann window STFT -> (257, T) real & imag spectral matrices
            val stftRes = stftEngine.stft(inputPcm)
            val T = stftRes.timeFrames

            // Flatten 2D matrices into 1D row-major buffers for (1, 257, T)
            val flatReal = FloatArray(257 * T)
            val flatImag = FloatArray(257 * T)

            for (f in 0 until 257) {
                for (t in 0 until T) {
                    val idx = f * T + t
                    flatReal[idx] = stftRes.real[f][t]
                    flatImag[idx] = stftRes.imag[f][t]
                }
            }

            // 2. Prepare ONNX Input Tensors
            val shapeSpec = longArrayOf(1, 257, T.toLong())
            val shapeHState = longArrayOf(2, 1, 128)

            val tensorReal = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatReal), shapeSpec)
            val tensorImag = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatImag), shapeSpec)
            val tensorHState = OnnxTensor.createTensor(env, FloatBuffer.wrap(currentHState), shapeHState)

            val inputMap = mapOf(
                "noisy_real" to tensorReal,
                "noisy_imag" to tensorImag,
                "h_state" to tensorHState
            )

            // 3. Execute Neural Inference
            val outputs = session?.run(inputMap)

            val enhRealTensor = outputs?.get(0)?.value as? Array<Array<FloatArray>>
            val enhImagTensor = outputs?.get(1)?.value as? Array<Array<FloatArray>>
            val nextHStateTensor = outputs?.get(4)?.value as? Array<Array<FloatArray>>

            // Extract updated h_state (2, 1, 128) for continuous streaming context across chunks
            if (nextHStateTensor != null && nextHStateTensor.size >= 2) {
                var hIdx = 0
                for (layer in 0 until 2) {
                    val row = nextHStateTensor[layer][0]
                    for (valItem in row) {
                        currentHState[hIdx++] = valItem
                    }
                }
            }

            // Clean up tensors
            tensorReal.close()
            tensorImag.close()
            tensorHState.close()

            if (enhRealTensor == null || enhImagTensor == null) {
                outputs?.close()
                return AncResult(inputPcm.clone(), null, false, "ONNX output tensors were null")
            }

            // Convert enh_real and enh_imag back to (257, T) matrices
            val enhRealMat = Array(257) { FloatArray(T) }
            val enhImagMat = Array(257) { FloatArray(T) }

            for (f in 0 until 257) {
                for (t in 0 until T) {
                    enhRealMat[f][t] = enhRealTensor[0][f][t]
                    enhImagMat[f][t] = enhImagTensor[0][f][t]
                }
            }

            outputs.close()

            // 4. Synthesize Enhanced Waveform via OLA iSTFT
            val enhancedPcm = stftEngine.istft(enhRealMat, enhImagMat, inputPcm.size)

            // 5. Calculate Genuine Level Reduction (10*log10(inEnergy / outEnergy))
            var inEnergy = 0.0
            var outEnergy = 0.0
            for (i in inputPcm.indices) {
                inEnergy += inputPcm[i] * inputPcm[i]
                outEnergy += enhancedPcm[i] * enhancedPcm[i]
            }

            val levelReductionDb = if (inEnergy > 1e-9 && outEnergy > 1e-9) {
                10.0 * Math.log10(inEnergy / outEnergy)
            } else {
                0.0
            }

            return AncResult(
                enhancedPcm = enhancedPcm,
                levelReductionDb = levelReductionDb,
                processed = true,
                failureReason = null
            )

        } catch (e: Throwable) {
            // Same reasoning as initOnnxSession: a native-side failure here must degrade to
            // storing the unprocessed audio, never take the capture service down.
            android.util.Log.e("ANC", "ONNX inference failed: " + e.message, e)
            return AncResult(
                enhancedPcm = inputPcm.clone(),
                levelReductionDb = null,
                processed = false,
                failureReason = e.javaClass.simpleName + ": " + (e.message ?: "no detail")
            )
        }
    }
}
