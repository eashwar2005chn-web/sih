package org.drdo.voicelogger.data.repo

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.drdo.voicelogger.data.db.AppDatabase
import org.drdo.voicelogger.data.model.AuditLog
import org.drdo.voicelogger.data.model.LogbookEntry
import org.drdo.voicelogger.storage.StorageLocationManager
import java.time.Instant
import java.util.UUID

/**
 * All destructive operations on the logbook go through here.
 *
 * A logbook entry is two things - a Room row and an encrypted file - and deleting only one of
 * them is a bug in either direction: an orphaned file is undiscoverable and un-deletable
 * through the UI, and an orphaned row points at audio that no longer exists. The file is
 * removed first, so the worst case is a row whose audio is already gone, which the app can
 * see and report.
 *
 * DELETION IS AUDITED, AND THAT IS NOT DECORATION. Removing an entry necessarily breaks the
 * prevSha256 -> sha256 link of the entry that followed it. Silently repairing that link by
 * rewriting the successor would make deletion undetectable, which defeats the entire purpose
 * of a tamper-evident chain. Instead every deletion writes an ENTRY_DELETED record carrying
 * the removed entry's own hash and its predecessor's, so [ChainVerifier] can bridge the gap
 * and distinguish "an operator deleted this on the record" from "someone altered the log".
 */
object LogbookRepository {

    const val EVENT_ENTRY_DELETED = "ENTRY_DELETED"

    data class DeleteResult(val deleted: Int, val fileFailures: Int)

    private fun auditRecord(entry: LogbookEntry, reason: String) = AuditLog(
        id = UUID.randomUUID().toString(),
        timestamp = Instant.now().toString(),
        eventType = EVENT_ENTRY_DELETED,
        // The two hashes are the bridge data. Format is parsed by ChainVerifier - keep in sync.
        details = "deleted=${entry.id} sha256=${entry.sha256} prev=${entry.prevSha256} " +
                "class=${entry.classification} sync=${entry.syncState} reason=$reason",
        sha256 = entry.sha256
    )

    suspend fun deleteEntry(
        ctx: Context,
        entry: LogbookEntry,
        reason: String = "operator_single"
    ): DeleteResult = withContext(Dispatchers.IO) {
        val db = AppDatabase.getDatabase(ctx)
        val fileGone = StorageLocationManager.deleteSegment(ctx, entry.filePath)
        db.auditLogDao().insertAuditLog(auditRecord(entry, reason))
        db.logbookEntryDao().deleteById(entry.id)
        DeleteResult(deleted = 1, fileFailures = if (fileGone) 0 else 1)
    }

    /**
     * Deletes every entry. Each one is audited individually rather than as a single summary
     * record, so the chain remains bridgeable afterwards.
     */
    suspend fun deleteAll(ctx: Context, reason: String = "operator_purge_all"): DeleteResult =
        withContext(Dispatchers.IO) {
            val db = AppDatabase.getDatabase(ctx)
            val all = db.logbookEntryDao().getAllEntriesAsc()
            var failures = 0
            for (e in all) {
                if (!StorageLocationManager.deleteSegment(ctx, e.filePath)) failures++
                db.auditLogDao().insertAuditLog(auditRecord(e, reason))
            }
            db.logbookEntryDao().deleteAllRows()
            DeleteResult(deleted = all.size, fileFailures = failures)
        }

    /**
     * Sums the sizes actually present on disk right now, which is not the same as summing the
     * sizeBytes column: entries purged after upload still have a row. The settings screen shows
     * the real occupied space, so this walks the files.
     */
    suspend fun bytesOnDisk(ctx: Context): Long = withContext(Dispatchers.IO) {
        AppDatabase.getDatabase(ctx).logbookEntryDao().getAllEntries()
            .sumOf { StorageLocationManager.sizeBytes(ctx, it.filePath) }
    }
}
