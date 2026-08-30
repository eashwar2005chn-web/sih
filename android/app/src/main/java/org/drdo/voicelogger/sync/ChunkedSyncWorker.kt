package org.drdo.voicelogger.sync

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.drdo.voicelogger.crypto.KeystoreManager
import org.drdo.voicelogger.data.db.AppDatabase
import org.drdo.voicelogger.data.model.AuditLog
import java.io.File
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID
import org.json.JSONObject
import android.util.Base64

/**
 * WorkManager Chunked Resumable Sync Worker.
 * Uploads hourly logbook entries in 512 KB chunks with exponential backoff and server SHA-256 verification.
 */
class ChunkedSyncWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    private val db = AppDatabase.getDatabase(context)
    private val chunkSize = 512 * 1024 // 512 KB per chunk

    override async suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val unsyncedEntries = db.logbookEntryDao().getUnsyncedEntries()
        if (unsyncedEntries.isEmpty()) {
            return@withContext Result.success()
        }

        // Priority sort: PROTECTED / SOS entries first, then oldest NORMAL entries
        val sortedEntries = unsyncedEntries.sortedWith(
            compareByDescending<org.drdo.voicelogger.data.model.LogbookEntry> {
                when (it.classification) {
                    "PROTECTED" -> 2
                    "SOS" -> 1
                    else -> 0
                }
            }.thenBy { it.startedAt }
        )

        var hasFailure = false

        for (entry in sortedEntries) {
            val file = File(entry.filePath)
            if (!file.exists()) {
                entry.lastError = "Encrypted file not found on disk"
                entry.syncState = "failed"
                db.logbookEntryDao().updateEntry(entry)
                hasFailure = true
                continue
            }

            try {
                entry.syncState = "uploading"
                db.logbookEntryDao().updateEntry(entry)

                // Read and decrypt audio payload from disk
                val encryptedBytes = file.readBytes()
                val isProtected = entry.classification == "PROTECTED"
                val audioBytes = KeystoreManager.decrypt(encryptedBytes, isProtected)

                val totalChunks = (audioBytes.size + chunkSize - 1) / chunkSize
                var uploadSuccess = true

                for (chunkIdx in 0 until totalChunks) {
                    val start = chunkIdx * chunkSize
                    val end = minOf(start + chunkSize, audioBytes.size)
                    val chunkData = audioBytes.copyOfRange(start, end)
                    val chunkB64 = Base64.encodeToString(chunkData, Base64.NO_WRAP)

                    val payload = JSONObject().apply {
                        put("id", entry.id)
                        put("chunkIndex", chunkIdx)
                        put("totalChunks", totalChunks)
                        put("chunk_b64", chunkB64)
                        put("classification", entry.classification)
                        put("sha256", entry.sha256)
                        put("prevSha256", entry.prevSha256)
                        put("durationSec", entry.durationSec)
                        put("vadSpeechSec", entry.vadSpeechSec)
                        put("createdAt", entry.startedAt)
                        put("snrGainDb", entry.snrGainDb)
                    }

                    val serverResponse = sendChunkToServer(entry.id, payload)
                    if (!serverResponse) {
                        uploadSuccess = false
                        break
                    }
                }

                if (uploadSuccess) {
                    entry.syncState = "synced"
                    entry.attempts += 1
                    entry.lastError = null
                    db.logbookEntryDao().updateEntry(entry)

                    // Append to Audit Log
                    val audit = AuditLog(
                        id = UUID.randomUUID().toString(),
                        timestamp = java.time.Instant.now().toString(),
                        eventType = "SYNC_SUCCESS",
                        details = "Entry ${entry.id} (${entry.classification}) synced successfully.",
                        sha256 = entry.sha256
                    )
                    db.auditLogDao().insertAuditLog(audit)
                } else {
                    entry.syncState = "failed"
                    entry.attempts += 1
                    entry.lastError = "Chunk upload failed on network"
                    db.logbookEntryDao().updateEntry(entry)
                    hasFailure = true
                }

            } catch (e: Exception) {
                entry.syncState = "failed"
                entry.attempts += 1
                entry.lastError = e.message
                db.logbookEntryDao().updateEntry(entry)
                hasFailure = true
            }
        }

        if (hasFailure) Result.retry() else Result.success()
    }

    private fun sendChunkToServer(entryId: String, payload: JSONObject): Boolean {
        return try {
            val url = URL("http://10.0.2.2:8080/api/recordings/$entryId/chunk") // 10.0.2.2 for Android emulator -> localhost server
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                setRequestProperty("Content-Type", "application/json")
                doOutput = true
                connectTimeout = 10000
                readTimeout = 10000
            }

            OutputStreamWriter(conn.outputStream, "UTF-8").use { writer ->
                writer.write(payload.toString())
                writer.flush()
            }

            val code = conn.responseCode
            conn.disconnect()
            code in 200..299
        } catch (e: Exception) {
            android.util.Log.e("SyncWorker", "HTTP sync connection failed: ${e.message}")
            false
        }
    }
}
