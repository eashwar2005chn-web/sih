package org.drdo.voicelogger.sync

import android.content.Context
import android.util.Base64
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.drdo.voicelogger.crypto.KeystoreManager
import org.drdo.voicelogger.data.db.AppDatabase
import org.drdo.voicelogger.data.model.AuditLog
import org.drdo.voicelogger.data.model.LogbookEntry
import org.drdo.voicelogger.data.prefs.AppSettings
import org.drdo.voicelogger.storage.StorageLocationManager
import org.json.JSONObject
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

/**
 * Uploads pending logbook entries to the configured cloud endpoint in 512 KB chunks.
 *
 * Scheduled by [SyncScheduler] - periodically, after each segment rotation, and on demand.
 * The network constraint lives on the WorkRequest, so this worker only ever runs when a
 * connection exists; that is what makes the offline-first requirement work without any polling.
 *
 * The destination is whatever the operator entered in settings. There is NO built-in fallback
 * URL: an unconfigured endpoint means the worker is never scheduled at all, rather than quietly
 * posting recordings somewhere the operator did not choose.
 */
class ChunkedSyncWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    companion object {
        const val PROGRESS_DONE = "done"
        const val PROGRESS_TOTAL = "total"
        const val KEY_UPLOADED = "uploaded"
        const val KEY_FAILED = "failed"
        const val KEY_MESSAGE = "message"
        private const val CHUNK_SIZE = 512 * 1024
    }

    private val db = AppDatabase.getDatabase(context)

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val ctx = applicationContext

        val base = AppSettings.getCloudEndpoint(ctx).trim().trimEnd('/')
        if (base.isEmpty()) {
            // Not an error. Nothing is configured, so there is nothing to do.
            return@withContext Result.success(
                workDataOf(KEY_MESSAGE to "No cloud endpoint configured")
            )
        }

        val pending = db.logbookEntryDao().getUnsyncedEntries()
        if (pending.isEmpty()) {
            return@withContext Result.success(
                workDataOf(KEY_UPLOADED to 0, KEY_MESSAGE to "Nothing to upload")
            )
        }

        // PROTECTED (duress) first, then SOS, then oldest normal traffic.
        val sorted = pending.sortedWith(
            compareByDescending<LogbookEntry> {
                when (it.classification) {
                    "PROTECTED" -> 2
                    "SOS" -> 1
                    else -> 0
                }
            }.thenBy { it.startedAt }
        )

        var uploaded = 0
        var failed = 0
        val deleteAfter = AppSettings.isDeleteAfterSync(ctx)

        for ((index, entry) in sorted.withIndex()) {
            if (isStopped) break
            setProgress(workDataOf(PROGRESS_DONE to index, PROGRESS_TOTAL to sorted.size))

            val stored = StorageLocationManager.readSegment(ctx, entry.filePath)
            if (stored == null) {
                entry.lastError = "Encrypted segment not found at " + entry.filePath
                entry.syncState = "failed"
                db.logbookEntryDao().updateEntry(entry)
                failed++
                continue
            }

            try {
                entry.syncState = "uploading"
                db.logbookEntryDao().updateEntry(entry)

                val audioBytes = KeystoreManager.decrypt(
                    stored, entry.classification == "PROTECTED"
                )
                val totalChunks = (audioBytes.size + CHUNK_SIZE - 1) / CHUNK_SIZE
                var ok = true

                for (chunkIdx in 0 until totalChunks) {
                    if (isStopped) {
                        ok = false
                        break
                    }
                    val start = chunkIdx * CHUNK_SIZE
                    val end = minOf(start + CHUNK_SIZE, audioBytes.size)
                    val chunkData = audioBytes.copyOfRange(start, end)

                    val payload = JSONObject()
                    payload.put("id", entry.id)
                    payload.put("deviceId", entry.deviceId)
                    payload.put("chunkIndex", chunkIdx)
                    payload.put("totalChunks", totalChunks)
                    payload.put("chunk_b64", Base64.encodeToString(chunkData, Base64.NO_WRAP))
                    payload.put("classification", entry.classification)
                    payload.put("sha256", entry.sha256)
                    payload.put("prevSha256", entry.prevSha256)
                    payload.put("durationSec", entry.durationSec)
                    payload.put("vadSpeechSec", entry.vadSpeechSec)
                    payload.put("createdAt", entry.startedAt)
                    payload.put("sampleRate", entry.sampleRate)
                    payload.put("codec", entry.codec)
                    if (entry.levelReductionDb != null) {
                        payload.put("levelReductionDb", entry.levelReductionDb)
                    }

                    if (!postChunk(base, entry.id, payload)) {
                        ok = false
                        break
                    }
                }

                if (ok) {
                    entry.syncState = "synced"
                    entry.attempts += 1
                    entry.lastError = null
                    db.logbookEntryDao().updateEntry(entry)
                    uploaded++

                    db.auditLogDao().insertAuditLog(
                        AuditLog(
                            id = UUID.randomUUID().toString(),
                            timestamp = java.time.Instant.now().toString(),
                            eventType = "SYNC_SUCCESS",
                            details = "Entry " + entry.id + " (" + entry.classification +
                                    ") uploaded to " + base,
                            sha256 = entry.sha256
                        )
                    )

                    // Opt-in only. Removing local audio right after upload makes the upload
                    // unverifiable afterwards, so the operator has to ask for it explicitly.
                    // The row and its hashes stay, so the chain still verifies by link.
                    if (deleteAfter && entry.classification == "NORMAL") {
                        StorageLocationManager.deleteSegment(ctx, entry.filePath)
                    }
                } else {
                    entry.syncState = "failed"
                    entry.attempts += 1
                    entry.lastError = "Chunk upload rejected or connection failed"
                    db.logbookEntryDao().updateEntry(entry)
                    failed++
                }
            } catch (e: Exception) {
                entry.syncState = "failed"
                entry.attempts += 1
                entry.lastError = e.message
                db.logbookEntryDao().updateEntry(entry)
                failed++
            }
        }

        val out = workDataOf(KEY_UPLOADED to uploaded, KEY_FAILED to failed)
        if (failed > 0) Result.retry() else Result.success(out)
    }

    /** POST one chunk to {base}/recordings/{id}/chunk. Returns true on a 2xx response. */
    private fun postChunk(base: String, entryId: String, payload: JSONObject): Boolean {
        var conn: HttpURLConnection? = null
        return try {
            val url = URL(base + "/recordings/" + entryId + "/chunk")
            conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.setRequestProperty("Content-Type", "application/json")
            conn.doOutput = true
            conn.connectTimeout = 15000
            conn.readTimeout = 30000

            OutputStreamWriter(conn.outputStream, "UTF-8").use { w ->
                w.write(payload.toString())
                w.flush()
            }
            val code = conn.responseCode
            if (code !in 200..299) {
                android.util.Log.w("SyncWorker", "Server returned HTTP " + code + " for " + url)
            }
            code in 200..299
        } catch (e: Exception) {
            android.util.Log.e("SyncWorker", "Upload failed: " + e.message)
            false
        } finally {
            conn?.disconnect()
        }
    }
}
