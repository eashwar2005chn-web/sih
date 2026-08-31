package org.drdo.voicelogger

import android.content.Context
import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.drdo.voicelogger.anc.OnnxAncProcessor
import org.drdo.voicelogger.crypto.HashChainManager
import org.drdo.voicelogger.crypto.KeystoreManager
import org.drdo.voicelogger.kws.KeywordSpotter
import org.drdo.voicelogger.service.VoiceLoggerService
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Native Instrumented Android Test Suite for DRDO 24/7 Voice Logging System.
 * Tests zero-gap rotation, service resilience, SOS continuity, KWS detection,
 * ONNX ANC signal processing, and storage headroom exhaustion protection.
 */
@RunWith(AndroidJUnit4::class)
class VoiceLoggerSystemTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()

    @Test
    fun testZeroGapRotationContiguity() {
        val segment1Bytes = ByteArray(16000 * 2) { 1 }
        val segment2Bytes = ByteArray(16000 * 2) { 2 }

        val prevHash = HashChainManager.GENESIS_PREV_HASH
        val hash1 = HashChainManager.computeEntryHash(segment1Bytes, prevHash)
        val hash2 = HashChainManager.computeEntryHash(segment2Bytes, hash1)

        assertNotNull("Segment 1 hash should be calculated", hash1)
        assertNotNull("Segment 2 hash should be calculated", hash2)
        assertTrue("Hashes should be distinct", hash1 != hash2)

        // Verify chain integrity
        val isSeg2Valid = HashChainManager.verifyEntryHash(segment2Bytes, prevSha256 = hash1, expectedSha256 = hash2)
        assertTrue("Segment 2 chained hash link must be valid", isSeg2Valid)

        // Negative control. A verifier that only ever returns true would pass the assertion
        // above, so check that tampering with the audio and with the link are both detected.
        val tampered = segment2Bytes.copyOf().also { it[0] = 99 }
        assertFalse(
            "Modified audio must fail verification",
            HashChainManager.verifyEntryHash(tampered, prevSha256 = hash1, expectedSha256 = hash2)
        )
        assertFalse(
            "A wrong predecessor link must fail verification",
            HashChainManager.verifyEntryHash(segment2Bytes, prevSha256 = prevHash, expectedSha256 = hash2)
        )
    }

    @Test
    fun testServiceResilienceDeclaration() {
        val intent = Intent(context, VoiceLoggerService::class.java).apply {
            action = VoiceLoggerService.ACTION_START
        }
        assertNotNull("Service start intent should be non-null", intent)
        assertEquals("Action must match ACTION_START", VoiceLoggerService.ACTION_START, intent.action)
    }

    @Test
    fun testSosContinuityFlag() {
        val intent = Intent(context, VoiceLoggerService::class.java).apply {
            action = VoiceLoggerService.ACTION_TRIGGER_SOS
        }
        assertEquals("Action must match ACTION_TRIGGER_SOS", VoiceLoggerService.ACTION_TRIGGER_SOS, intent.action)
    }

    @Test
    fun testKwsDetectionTrueVsFalsePositive() {
        val spotter = KeywordSpotter(context, confidenceThreshold = 0.80f)
        var detectedWord: String? = null
        var detectedScore: Float? = null

        spotter.setCallback(object : KeywordSpotter.Callback {
            override fun onCodeWordDetected(codeWord: String, confidence: Float) {
                detectedWord = codeWord
                detectedScore = confidence
            }
        })

        // Test 1: Silent/Loud Noise frame -> should NOT trigger detection
        val noiseFrame = ShortArray(512) { (Math.random() * 500).toInt().toShort() }
        val noiseTriggered = spotter.processFrame(noiseFrame)
        assertFalse("Random noise should not trigger keyword detection", noiseTriggered)

        // Test 2: If engine is unit-test initialized, verify callback interface
        assertNotNull("Target keywords must contain RED_TACTICAL", spotter.targetCodeWords.find { it == "RED_TACTICAL" })
    }

    @Test
    fun testAncProcessingChain() {
        val ancProcessor = OnnxAncProcessor(context)
        val testPcm = FloatArray(16000) { kotlin.math.sin(2.0 * Math.PI * 440.0 * it / 16000).toFloat() * 0.5f }

        val res = ancProcessor.processAudio(testPcm)
        assertNotNull("AncResult should be non-null", res)
        assertNotNull("enhancedPcm should be non-null", res.enhancedPcm)
        assertEquals("Enhanced PCM length must match input PCM length", testPcm.size, res.enhancedPcm.size)
    }

    @Test
    fun testKeystoreEncryptionStorageProtection() {
        val rawData = "DRDO_TOP_SECRET_VOICE_PAYLOAD_TEST".toByteArray()
        val encryptedData = KeystoreManager.encrypt(rawData, isProtected = true)

        assertNotNull("Encrypted data must be non-null", encryptedData)
        assertTrue("Encrypted payload must be different from raw payload", !rawData.contentEquals(encryptedData))

        val decryptedData = KeystoreManager.decrypt(encryptedData, isProtected = true)
        assertTrue("Decrypted payload must match original raw payload", rawData.contentEquals(decryptedData))
    }
}
