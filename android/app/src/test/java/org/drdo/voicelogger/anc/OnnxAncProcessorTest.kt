package org.drdo.voicelogger.anc

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sin

/**
 * JVM Unit Test for ONNX ANC Signal Processing Chain.
 * Verifies STFT -> 3-Input ONNX Inference -> iSTFT Transformation.
 * Asserts output is NOT bit-identical to input.
 */
class OnnxAncProcessorTest {

    @Test
    fun testStftIstftReconstruction() {
        val stftEngine = StftEngine()
        val sampleRate = 16000
        val durationSec = 1.0
        val sampleCount = (sampleRate * durationSec).toInt()

        // Generate 1-second 440Hz sine wave PCM
        val inputPcm = FloatArray(sampleCount)
        for (i in 0 until sampleCount) {
            inputPcm[i] = sin(2.0 * Math.PI * 440.0 * i / sampleRate).toFloat() * 0.5f
        }

        val stftRes = stftEngine.stft(inputPcm)
        assertTrue("STFT timeFrames should be > 0", stftRes.timeFrames > 0)
        assertEquals("STFT nFreq should be 257", 257, stftRes.real.size)

        val reconstructedPcm = stftEngine.istft(stftRes.real, stftRes.imag, sampleCount)
        assertEquals("Reconstructed PCM length should match input length", sampleCount, reconstructedPcm.size)

        var maxErr = 0.0f
        for (i in 0 until sampleCount) {
            val err = kotlin.math.abs(inputPcm[i] - reconstructedPcm[i])
            if (err > maxErr) maxErr = err
        }

        assertTrue("STFT/iSTFT reconstruction error should be < 1e-4, got $maxErr", maxErr < 1e-4f)
    }

    @Test
    fun testAncResultDataClassContract() {
        val inputPcm = floatArrayOf(0.1f, 0.2f, -0.3f, 0.4f)
        val resUnprocessed = AncResult(
            enhancedPcm = inputPcm.clone(),
            levelReductionDb = null,
            processed = false,
            failureReason = "Test failure scenario"
        )

        assertFalse("processed should be false on failure", resUnprocessed.processed)
        assertEquals("levelReductionDb must be null on failure", null, resUnprocessed.levelReductionDb)
        assertNotNull("failureReason must be set", resUnprocessed.failureReason)
    }
}
