package org.drdo.voicelogger.data.model

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "logbook_entries")
data class LogbookEntry(
    @PrimaryKey
    val id: String,                  // UUID v4
    val startedAt: String,           // ISO-8601 Timestamp
    val endedAt: String,             // ISO-8601 Timestamp
    val durationSec: Double,
    val filePath: String,            // Path to encrypted audio file in app-private storage
    val sizeBytes: Long,
    // The app writes raw little-endian 16-bit PCM. This field used to default to
    // "Opus_24kbps", which was never true - no Opus encoder exists anywhere in the codebase -
    // and it was uploaded to the server on every chunk as if it were fact.
    // At 16 kHz mono this is 32 KB/s, i.e. ~115 MB per hour of capture.
    val codec: String = "PCM_S16LE",
    val sampleRate: Int = 16000,
    val deviceId: String = "ANDROID_FIELD_UNIT_01",
    var syncState: String = "local",  // "local" | "queued" | "uploading" | "synced" | "failed"
    var serverId: String? = null,
    val sha256: String,              // Hash of this entry's audio + prevSha256
    val prevSha256: String,          // Hash of preceding entry in chain
    val vadSpeechSec: Double,        // Actual speech time recorded after VAD gating
    var attempts: Int = 0,
    var lastError: String? = null,
    var classification: String = "NORMAL", // "NORMAL" | "SOS" | "PROTECTED"
    val levelReductionDb: Double? = null   // Genuinely measured Level Reduction in dB (null if unprocessed)
)
