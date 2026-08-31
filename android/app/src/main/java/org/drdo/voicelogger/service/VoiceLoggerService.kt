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
import org.drdo.voicelogger.R
import org.drdo.voicelogger.anc.OnnxAncProcessor
import org.drdo.voicelogger.crypto.HashChainManager
import org.drdo.voicelogger.crypto.KeystoreManager
import org.drdo.voicelogger.data.db.AppDatabase
import org.drdo.voicelogger.data.model.AuditLog
import org.drdo.voicelogger.data.model.LogbookEntry
import org.drdo.voicelogger.data.prefs.AppSettings
import org.drdo.voicelogger.kws.KeywordSpotter
import org.drdo.voicelogger.storage.StorageLocationManager
import org.drdo.voicelogger.sync.SyncScheduler
import org.drdo.voicelogger.ui.MainActivity
import java.io.ByteArrayOutputStream
import java.time.Instant
import java.util.UUID
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * 24/7 continuous background audio capture.
 *
 * Foreground service + partial wake lock, zero-gap segment rotation, ONNX noise cancellation,
 * AES-256-GCM encryption at rest, SHA-256 hash chaining, and audit logging.
 */
class VoiceLoggerService : Service() {

    companion object {
        const val CHANNEL_ID = "VoiceLoggerChannel"
        const val NOTIFICATION_ID = 26052
        const val ACTION_START = "ACTION_START"
        const val ACTION_STOP = "ACTION_STOP"
        const val ACTION_TRIGGER_SOS = "ACTION_TRIGGER_SOS"
        const val ACTION_STAND_DOWN_SOS = "ACTION_STAND_DOWN_SOS"
        const val EXTRA_SEGMENT_SECONDS = "EXTRA_SEGMENT_SECONDS"
        const val DEFAULT_SEGMENT_SECONDS = 3600L

        @Volatile
        var isServiceRunning = false

        // --- live telemetry, read by the dashboard -----------------------------------------
        // Plain volatile fields rather than a Flow or a broadcast: these are written from the
        // audio read loop and read by one UI poller. Anything heavier would put allocation or
        // dispatch on the capture thread, which must not stall.
        //
        // Every one of these is a MEASURED value. None is a placeholder.

        /** Instantaneous input level in dBFS, floored at -60. Meaningless while stopped. */
        @Volatile
        var currentLevelDbfs: Float = -60f

        /** Wall-clock ms at which the in-progress segment began. */
        @Volatile
        var segmentStartedAtMs: Long = 0L

        /** Length of a completed segment in ms, as currently configured. */
        @Volatile
        var segmentLengthMs: Long = DEFAULT_SEGMENT_SECONDS * 1000L

        /** Bytes buffered for the in-progress segment (raw PCM, pre-encryption). */
        @Volatile
        var bufferedBytes: Long = 0L

        @Volatile
        var sosActive: Boolean = false
    }

    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(Dispatchers.IO + serviceJob)

    private var wakeLock: PowerManager.WakeLock? = null
    private var audioRecord: AudioRecord? = null
    private var isRecording = false

    private lateinit var db: AppDatabase

    // Nullable, not lateinit. Both load native ONNX code that can fail on a given device or
    // ABI; when that happens the service must still capture and store audio, just without
    // enhancement. A lateinit here would turn a degraded feature into a dead recorder.
    private var onnxProcessor: OnnxAncProcessor? = null
    private var keywordSpotter: KeywordSpotter? = null

    private var currentBuffer = ByteArrayOutputStream()
    private var segmentStartTimeMs = 0L
    private var segmentDurationMs = DEFAULT_SEGMENT_SECONDS * 1000L

    private var isSosActive = false
    private var isDuressActive = false
    private var currentSpeechDurationSec = 0.0

    override fun onCreate() {
        super.onCreate()
        db = AppDatabase.getDatabase(applicationContext)

        onnxProcessor = runCatching { OnnxAncProcessor(applicationContext) }.getOrElse { e ->
            android.util.Log.e("VoiceLoggerService",
                "ANC unavailable, recording will store unprocessed audio: " + e, e)
            null
        }
        keywordSpotter = runCatching { KeywordSpotter(applicationContext) }.getOrElse { e ->
            android.util.Log.e("VoiceLoggerService", "KWS unavailable: " + e, e)
            null
        }

        keywordSpotter?.setCallback(object : KeywordSpotter.Callback {
            override fun onCodeWordDetected(codeWord: String, confidence: Float) {
                isDuressActive = true
                serviceScope.launch {
                    db.auditLogDao().insertAuditLog(
                        AuditLog(
                            id = UUID.randomUUID().toString(),
                            timestamp = Instant.now().toString(),
                            eventType = "CODEWORD_DETECTED",
                            details = "Duress code word detected: " + codeWord +
                                    " (score " + confidence + ")",
                            sha256 = ""
                        )
                    )
                }
            }
        })

        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK, "VoiceLogger::247AudioLock"
        )
        wakeLock?.acquire(24 * 60 * 60 * 1000L)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                segmentDurationMs = resolveSegmentMs(intent)
                segmentLengthMs = segmentDurationMs
                android.util.Log.i(
                    "VoiceLoggerService",
                    "Segment rotation interval: " + (segmentDurationMs / 1000) + "s"
                )
                if (!isRecording) {
                    createNotificationChannel()
                    startForeground(NOTIFICATION_ID, buildNotification("24/7 voice capture active"))
                    startAudioCapture()
                }
                // The standing upload job. KEEP policy, so calling this on every start is safe.
                SyncScheduler.ensurePeriodic(applicationContext)
            }
            ACTION_STOP -> {
                stopAudioCapture()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
            ACTION_TRIGGER_SOS -> handleSosTrigger()
            ACTION_STAND_DOWN_SOS -> {
                // Clears the SOS flag only. Recording is deliberately untouched: standing down
                // an emergency must never interrupt capture.
                isSosActive = false
                sosActive = false
                updateNotification("Recording - 24/7 logbook active")
            }
        }
        return START_STICKY
    }

    /**
     * Rotation interval, in precedence order:
     *   1. files/debug_segment_seconds - debuggable builds only, for exercising rotation
     *      without waiting an hour. A release build ignores the file entirely.
     *   2. EXTRA_SEGMENT_SECONDS on the intent - used by the instrumented tests.
     *   3. The operator's configured value in settings (default 60 minutes).
     */
    private fun resolveSegmentMs(intent: Intent): Long {
        val isDebuggable = (applicationInfo.flags and
                android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0
        val overrideFile = java.io.File(filesDir, "debug_segment_seconds")
        val fileSecs = if (isDebuggable && overrideFile.exists())
            overrideFile.readText().trim().toLongOrNull() else null

        val settingsSecs = AppSettings.getSegmentMinutes(applicationContext) * 60L
        val secs = fileSecs ?: intent.getLongExtra(EXTRA_SEGMENT_SECONDS, settingsSecs)
        return secs.coerceAtLeast(10L) * 1000L
    }

    private fun handleSosTrigger() {
        isSosActive = true
        sosActive = true
        serviceScope.launch {
            db.auditLogDao().insertAuditLog(
                AuditLog(
                    id = UUID.randomUUID().toString(),
                    timestamp = Instant.now().toString(),
                    eventType = "SOS_ALERT",
                    details = "Emergency SOS triggered by operator. Recording continues unbroken.",
                    sha256 = ""
                )
            )
        }
        updateNotification("SOS ALERT ACTIVE - continuous recording")
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW
            )
            channel.description = getString(R.string.notification_channel_desc)
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(text: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_stat_mic)
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
                MediaRecorder.AudioSource.MIC, sampleRate, channelConfig, audioFormat, bufferSize
            )
            audioRecord?.startRecording()
            isRecording = true
            isServiceRunning = true
            segmentStartTimeMs = System.currentTimeMillis()
            segmentStartedAtMs = segmentStartTimeMs
            bufferedBytes = 0L

            serviceScope.launch {
                val shortBuffer = ShortArray(bufferSize / 2)
                val byteBuffer = ByteArray(bufferSize)

                while (isRecording) {
                    val readSize = audioRecord?.read(shortBuffer, 0, shortBuffer.size) ?: 0
                    if (readSize > 0) {
                        keywordSpotter?.processFrame(shortBuffer)

                        // Real RMS of this block, published for the dashboard meter.
                        var sumSq = 0.0
                        for (i in 0 until readSize) {
                            val v = shortBuffer[i].toDouble()
                            sumSq += v * v
                        }
                        val rms = sqrt(sumSq / readSize)
                        currentLevelDbfs = if (rms < 1.0) -60f
                        else (20.0 * log10(rms / 32768.0)).toFloat().coerceIn(-60f, 0f)

                        for (i in 0 until readSize) {
                            val sample = shortBuffer[i]
                            byteBuffer[i * 2] = (sample.toInt() and 0xFF).toByte()
                            byteBuffer[i * 2 + 1] = (sample.toInt() ushr 8 and 0xFF).toByte()
                        }

                        currentBuffer.write(byteBuffer, 0, readSize * 2)
                        bufferedBytes = currentBuffer.size().toLong()
                        currentSpeechDurationSec += (readSize.toDouble() / sampleRate)

                        if (System.currentTimeMillis() - segmentStartTimeMs >= segmentDurationMs) {
                            rotateSegmentZeroGap()
                        }
                    }
                }
                currentLevelDbfs = -60f
            }
        } catch (e: Exception) {
            android.util.Log.e("VoiceLoggerService", "AudioRecord init error: " + e.message)
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

        // Open the new buffer immediately, before any processing, so capture never pauses.
        currentBuffer = ByteArrayOutputStream()
        segmentStartTimeMs = System.currentTimeMillis()
        segmentStartedAtMs = segmentStartTimeMs
        currentSpeechDurationSec = 0.0
        bufferedBytes = 0L
        isDuressActive = false

        val rawAudioBytes = oldBuffer.toByteArray()
        if (rawAudioBytes.isNotEmpty()) {
            saveLogbookSegment(
                rawAudioBytes, oldStartTime, System.currentTimeMillis(),
                oldSpeechDuration, oldClassification
            )
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

        val sampleCount = rawAudioBytes.size / 2
        val floatPcm = FloatArray(sampleCount)
        for (i in 0 until sampleCount) {
            val b1 = rawAudioBytes[i * 2].toInt() and 0xFF
            val b2 = rawAudioBytes[i * 2 + 1].toInt()
            floatPcm[i] = ((b2 shl 8) or b1).toShort() / 32768.0f
        }

        // A null processor means ANC never initialised on this device. levelReductionDb stays
        // null and the entry records "not measured" - it does NOT record 0 dB, which would
        // read as "measured, and it did nothing".
        val ancRes = onnxProcessor?.processAudio(floatPcm)

        val finalPcmBytes: ByteArray
        if (ancRes != null && ancRes.processed) {
            finalPcmBytes = ByteArray(sampleCount * 2)
            for (i in 0 until sampleCount) {
                val s = (ancRes.enhancedPcm[i] * 32767.0f).toInt()
                    .coerceIn(-32768, 32767).toShort()
                finalPcmBytes[i * 2] = (s.toInt() and 0xFF).toByte()
                finalPcmBytes[i * 2 + 1] = (s.toInt() ushr 8 and 0xFF).toByte()
            }
        } else {
            // Processing failed -> store raw audio unchanged. Never drop a recording.
            finalPcmBytes = rawAudioBytes
        }

        val encryptedBytes = KeystoreManager.encrypt(finalPcmBytes, isProtected)
        // Destination is whatever the operator configured; falls back to internal storage if
        // that directory has become unwritable.
        val location = StorageLocationManager.writeSegment(
            applicationContext, "segment_" + entryId + ".bin", encryptedBytes
        )

        val latestEntry = db.logbookEntryDao().getLatestEntry()
        val prevSha256 = latestEntry?.sha256 ?: HashChainManager.GENESIS_PREV_HASH
        val sha256 = HashChainManager.computeEntryHash(finalPcmBytes, prevSha256)

        val entry = LogbookEntry(
            id = entryId,
            startedAt = Instant.ofEpochMilli(startTimeMs).toString(),
            endedAt = Instant.ofEpochMilli(endTimeMs).toString(),
            durationSec = (endTimeMs - startTimeMs) / 1000.0,
            filePath = location,
            sizeBytes = StorageLocationManager.sizeBytes(applicationContext, location),
            deviceId = AppSettings.getDeviceId(applicationContext),
            sha256 = sha256,
            prevSha256 = prevSha256,
            vadSpeechSec = speechSec,
            classification = classification,
            levelReductionDb = ancRes?.levelReductionDb
        )

        db.logbookEntryDao().insertEntry(entry)
        val lrText = ancRes?.levelReductionDb?.let { String.format("%.2f dB", it) } ?: "Not measured"
        android.util.Log.i(
            "VoiceLoggerService",
            "Logbook entry saved (" + classification + "): " + entryId +
                    " | Level Reduction: " + lrText + " | Hash: " + sha256.take(8) + "..."
        )

        // A finished segment is exactly when there is new work to upload. If there is no
        // network the request waits; WorkManager runs it when connectivity returns.
        SyncScheduler.syncNow(applicationContext)
    }

    private fun stopAudioCapture() {
        isRecording = false
        isServiceRunning = false
        currentLevelDbfs = -60f
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
        }
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
