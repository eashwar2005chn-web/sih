package org.drdo.voicelogger.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import org.drdo.voicelogger.anc.OnnxAncProcessor
import org.drdo.voicelogger.crypto.HashChainManager
import org.drdo.voicelogger.crypto.KeystoreManager
import org.drdo.voicelogger.data.db.AppDatabase
import org.drdo.voicelogger.data.model.AuditLog
import org.drdo.voicelogger.data.model.LogbookEntry
import org.drdo.voicelogger.kws.KeywordSpotter
import org.drdo.voicelogger.ui.MainActivity
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.time.Instant
import java.util.UUID

/**
 * 24/7 Continuous Background Audio Capture Service.
 * Runs as Foreground Service with WakeLock.
 * Performs zero-gap hourly logbook segment rotation, ONNX noise cancellation,
 * AES-256-GCM encryption at rest, and audit trail logging.
 */
class VoiceLoggerService : Service() {

    companion object {
        const val CHANNEL_ID = "VoiceLoggerChannel"
        const val NOTIFICATION_ID = 26052
        const val ACTION_START = "ACTION_START"
        const val ACTION_STOP = "ACTION_STOP"
        const val ACTION_TRIGGER_SOS = "ACTION_TRIGGER_SOS"
        const val ACTION_STAND_DOWN_SOS = "ACTION_STAND_DOWN_SOS"

        @Volatile
        var isServiceRunning = false
    }

    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(Dispatchers.IO + serviceJob)

    private var wakeLock: PowerManager.WakeLock? = null
    private var audioRecord: AudioRecord? = null
    private var isRecording = false

    private lateinit var db: AppDatabase
    private lateinit var onnxProcessor: OnnxAncProcessor
    private lateinit var keywordSpotter: KeywordSpotter

    private var currentBuffer = ByteArrayOutputStream()
    private var segmentStartTimeMs = 0L
    private val segmentDurationMs = 3600 * 1000L // 1 Hour

    private var isSosActive = false
    private var isDuressActive = false
    private var currentSpeechDurationSec = 0.0

    override fun onCreate() {
        super.onCreate()
        db = AppDatabase.getDatabase(applicationContext)
        onnxProcessor = OnnxAncProcessor(applicationContext)
        keywordSpotter = KeywordSpotter(applicationContext)

        keywordSpotter.setCallback(object : KeywordSpotter.Callback {
            override fun onCodeWordDetected(codeWord: String, confidence: Float) {
                isDuressActive = true
                // Log to audit trail silently with engine name, model version, and real score
                serviceScope.launch {
                    val audit = AuditLog(
                        id = UUID.randomUUID().toString(),
                        timestamp = Instant.now().toString(),
                        eventType = "CODEWORD_DETECTED",
                        details = "Duress Code Word '$codeWord' detected by engine ONNX_KWS_v1.0 (Real Score: $confidence)",
                        sha256 = ""
                    )
                    db.auditLogDao().insertAuditLog(audit)
                }
            }
        })

        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "VoiceLogger::247AudioLock")
        wakeLock?.acquire(24 * 60 * 60 * 1000L) // Hold lock
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                if (!isRecording) {
                    createNotificationChannel()
                    startForeground(NOTIFICATION_ID, buildNotification("24/7 Voice Capture Active"))
                    startAudioCapture()
                }
            }
            ACTION_STOP -> {
                stopAudioCapture()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
            ACTION_TRIGGER_SOS -> {
                handleSosTrigger()
            }
            ACTION_STAND_DOWN_SOS -> {
                // Clears the SOS flag only. Recording is deliberately untouched: standing
                // down an emergency must never interrupt capture.
                isSosActive = false
                updateNotification("Recording — 24/7 logbook active")
            }
        }
        return START_STICKY
    }

    private fun handleSosTrigger() {
        isSosActive = true
        serviceScope.launch {
            val audit = AuditLog(
                id = UUID.randomUUID().toString(),
                timestamp = Instant.now().toString(),
                eventType = "SOS_ALERT",
                details = "Emergency SOS Alert triggered by operator. Recording continues unbroken.",
                sha256 = ""
            )
            db.auditLogDao().insertAuditLog(audit)
        }
        updateNotification("🚨 SOS ALERT ACTIVE — Continuous Recording")
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "24/7 Voice Logger Foreground Service",
                NotificationManager.IMPORTANCE_LOW
            )
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(text: String): Notification {
        val notificationIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, notificationIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("DRDO 24/7 Voice Logger")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun updateNotification(text: String) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID, buildNotification(text))
    }

    private fun startAudioCapture() {
        val sampleRate = 16000
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT
        val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat) * 2

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize
            )

            audioRecord?.startRecording()
            isRecording = true
            isServiceRunning = true
            segmentStartTimeMs = System.currentTimeMillis()

            serviceScope.launch {
                val shortBuffer = ShortArray(bufferSize / 2)
                val byteBuffer = ByteArray(bufferSize)

                while (isRecording) {
                    val readSize = audioRecord?.read(shortBuffer, 0, shortBuffer.size) ?: 0
                    if (readSize > 0) {
                        // 1. Process for duress code word via ONNX KWS
                        keywordSpotter.processFrame(shortBuffer)

                        // Convert short buffer to raw byte array
                        for (i in 0 until readSize) {
                            val sample = shortBuffer[i]
                            byteBuffer[i * 2] = (sample.toInt() and 0xFF).toByte()
                            byteBuffer[i * 2 + 1] = (sample.toInt() ushr 8 and 0xFF).toByte()
                        }

                        // Accumulate raw audio in segment buffer
                        currentBuffer.write(byteBuffer, 0, readSize * 2)
                        currentSpeechDurationSec += (readSize.toDouble() / sampleRate)

                        // Zero-gap hourly segment rotation check
                        if (System.currentTimeMillis() - segmentStartTimeMs >= segmentDurationMs) {
                            rotateSegmentZeroGap()
                        }
                    }
                }
            }
        } catch (e: Exception) {
            android.util.Log.e("VoiceLoggerService", "AudioRecord initialization error: ${e.message}")
        }
    }

    private suspend fun rotateSegmentZeroGap() {
        val oldBuffer = currentBuffer
        val oldStartTime = segmentStartTimeMs
        val oldSpeechDuration = currentSpeechDurationSec
        val oldClassification = when {
            isDuressActive -> "PROTECTED"
            isSosActive -> "SOS"
            else -> "NORMAL"
        }

        // Open new buffer immediately (zero gap)
        currentBuffer = ByteArrayOutputStream()
        segmentStartTimeMs = System.currentTimeMillis()
        currentSpeechDurationSec = 0.0
        isDuressActive = false

        // Process and persist previous completed segment asynchronously
        val rawAudioBytes = oldBuffer.toByteArray()
        if (rawAudioBytes.isNotEmpty()) {
            saveLogbookSegment(rawAudioBytes, oldStartTime, System.currentTimeMillis(), oldSpeechDuration, oldClassification)
        }
    }

    private suspend fun saveLogbookSegment(
        rawAudioBytes: ByteArray,
        startTimeMs: Long,
        endTimeMs: Long,
        speechSec: Double,
        classification: String
    ) {
        val entryId = UUID.randomUUID().toString()
        val isProtected = classification == "PROTECTED"

        // Convert raw bytes to FloatArray PCM for ONNX Noise Cancellation
        val sampleCount = rawAudioBytes.size / 2
        val floatPcm = FloatArray(sampleCount)
        for (i in 0 until sampleCount) {
            val b1 = rawAudioBytes[i * 2].toInt() and 0xFF
            val b2 = rawAudioBytes[i * 2 + 1].toInt()
            val sample = ((b2 shl 8) or b1).toShort()
            floatPcm[i] = sample / 32768.0f
        }

        // Run ONNX ANC signal chain (STFT -> 3 Named Inputs -> Streaming h_state -> iSTFT)
        val ancRes = onnxProcessor.processAudio(floatPcm)

        val finalPcmBytes: ByteArray
        if (ancRes.processed) {
            // Enhanced PCM audio
            finalPcmBytes = ByteArray(sampleCount * 2)
            for (i in 0 until sampleCount) {
                val s = (ancRes.enhancedPcm[i] * 32767.0f).toInt().coerceIn(-32768, 32767).toShort()
                finalPcmBytes[i * 2] = (s.toInt() and 0xFF).toByte()
                finalPcmBytes[i * 2 + 1] = (s.toInt() ushr 8 and 0xFF).toByte()
            }
        } else {
            // Processing failed -> store raw audio unchanged
            finalPcmBytes = rawAudioBytes
        }

        // Encrypt audio payload at rest via Keystore
        val encryptedBytes = KeystoreManager.encrypt(finalPcmBytes, isProtected)
        val file = File(filesDir, "segment_$entryId.bin")
        FileOutputStream(file).use { it.write(encryptedBytes) }

        // Fetch preceding hash in chain
        val latestEntry = db.logbookEntryDao().getLatestEntry()
        val prevSha256 = latestEntry?.sha256 ?: HashChainManager.GENESIS_PREV_HASH
        val sha256 = HashChainManager.computeEntryHash(finalPcmBytes, prevSha256)

        val durationSec = (endTimeMs - startTimeMs) / 1000.0

        val entry = LogbookEntry(
            id = entryId,
            startedAt = Instant.ofEpochMilli(startTimeMs).toString(),
            endedAt = Instant.ofEpochMilli(endTimeMs).toString(),
            durationSec = durationSec,
            filePath = file.absolutePath,
            sizeBytes = file.length(),
            sha256 = sha256,
            prevSha256 = prevSha256,
            vadSpeechSec = speechSec,
            classification = classification,
            levelReductionDb = ancRes.levelReductionDb
        )

        db.logbookEntryDao().insertEntry(entry)
        android.util.Log.i(
            "VoiceLoggerService",
            "Logbook Entry saved ($classification): $entryId | Level Reduction: ${ancRes.levelReductionDb?.let { String.format("%.2f dB", it) } ?: "Not Measured"} | Hash: ${sha256.take(8)}..."
        )
    }

    private fun stopAudioCapture() {
        isRecording = false
        isServiceRunning = false
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {}
        audioRecord = null
    }

    override fun onDestroy() {
        stopAudioCapture()
        wakeLock?.let { if (it.isHeld) it.release() }
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
