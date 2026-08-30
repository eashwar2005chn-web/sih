package org.drdo.voicelogger.data.model

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "audit_logs")
data class AuditLog(
    @PrimaryKey
    val id: String,                  // UUID v4
    val timestamp: String,           // ISO-8601 Timestamp
    val eventType: String,           // "SOS_ALERT" | "CODEWORD_DETECTED" | "SYNC_SUCCESS" | "PURGE" | "CONFIG_CHANGE" | "PERMISSION_GRANTED"
    val details: String,
    val sha256: String               // Audit log chain hash
)
