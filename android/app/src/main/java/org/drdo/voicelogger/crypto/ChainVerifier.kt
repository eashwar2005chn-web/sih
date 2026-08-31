package org.drdo.voicelogger.crypto

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.drdo.voicelogger.data.db.AppDatabase
import org.drdo.voicelogger.data.repo.LogbookRepository
import org.drdo.voicelogger.storage.StorageLocationManager

/**
 * Verifies the logbook hash chain for real.
 *
 * This exists because the UI previously displayed the fixed string
 * "SHA-256 Hash Chain: Active and Validated" from the layout XML. Nothing computed it. It
 * would have read "Validated" over an empty database and over a chain broken at entry 1
 * alike, which makes it worse than showing nothing at all: it is a tamper-evidence claim that
 * cannot detect tampering.
 *
 * Three independent properties are checked, and reported separately because conflating them
 * would overstate what was actually verified:
 *
 *   LINK    entry[i].prevSha256 == entry[i-1].sha256
 *           Cheap, database only. Catches deletion or reordering of entries.
 *
 *   CONTENT sha256(decrypt(file_on_disk) || prevSha256) == entry[i].sha256
 *           Needs the audio. Catches modification of the recording itself.
 *
 *   BRIDGE  A link gap explained by an audited ENTRY_DELETED record is not tampering. Operator
 *           deletions are legitimate and expected; what matters is that they are ON THE RECORD.
 *           An unexplained gap is still reported as broken.
 */
object ChainVerifier {

    /**
     * Ceiling on how much audio one verification pass will decrypt.
     *
     * An hour of 16 kHz PCM is ~115 MB, and decryption holds the ciphertext and the plaintext
     * in memory at once, so a full day of capture would be several GB of transient allocation
     * and would OOM long before it finished. Content verification therefore runs newest-first
     * until this budget is spent; everything past it is link-verified only and REPORTED AS
     * SUCH. Degrading to a weaker check is acceptable. Reporting a weaker check as a full one
     * is not.
     *
     * Streaming the digest through CipherInputStream would remove the ceiling, but Android's
     * CipherInputStream is known to swallow AEADBadTagException and return a short read rather
     * than throwing - which would make a tampered segment verify clean. A bounded one-shot
     * decrypt that definitely throws is the safer trade for a tamper-evidence claim.
     */
    private const val CONTENT_BUDGET_BYTES = 192L * 1024 * 1024

    /** No single entry above this is content-checked, whatever budget remains. */
    private const val MAX_SINGLE_ENTRY_BYTES = 96L * 1024 * 1024

    sealed class Status {
        /** No entries recorded yet. There is nothing to validate - say so, do not claim success. */
        object Empty : Status()

        data class Verified(
            val total: Int,
            val contentVerified: Int,
            /** Link verified, but the audio is gone (purged after upload, or storage moved). */
            val audioAbsent: Int,
            /** Audio present, but skipped to stay inside the memory budget. */
            val notChecked: Int,
            val bridgedDeletions: Int
        ) : Status()

        data class Broken(
            val position: Int,      // 1-based position in the chain
            val total: Int,
            val entryId: String,
            val reason: String
        ) : Status()
    }

    /** prevSha256 -> sha256 for every entry removed through an audited deletion. */
    private suspend fun deletionBridges(ctx: Context): Map<String, String> {
        val logs = AppDatabase.getDatabase(ctx).auditLogDao().getAllAuditLogs()
        val out = HashMap<String, String>()
        for (l in logs) {
            if (l.eventType != LogbookRepository.EVENT_ENTRY_DELETED) continue
            val sha = Regex("""sha256=([0-9a-fA-F]{64})""").find(l.details)?.groupValues?.get(1)
            val prev = Regex("""prev=(\S+)""").find(l.details)?.groupValues?.get(1)
            if (sha != null && prev != null) out[prev] = sha
        }
        return out
    }

    suspend fun verify(ctx: Context): Status = withContext(Dispatchers.IO) {
        val db = AppDatabase.getDatabase(ctx)
        val entries = db.logbookEntryDao().getAllEntriesAsc()
        if (entries.isEmpty()) return@withContext Status.Empty

        val bridges = deletionBridges(ctx)

        // Choose which entries get the full content check: newest first, until the budget is
        // spent. The most recent recordings are the ones most worth proving intact.
        val contentSet = HashSet<String>()
        var budget = CONTENT_BUDGET_BYTES
        for (e in entries.asReversed()) {
            val sz = e.sizeBytes
            if (sz > MAX_SINGLE_ENTRY_BYTES || sz > budget) continue
            contentSet.add(e.id)
            budget -= sz
        }

        var contentVerified = 0
        var audioAbsent = 0
        var notChecked = 0
        var bridged = 0
        var expectedPrev = HashChainManager.GENESIS_PREV_HASH

        entries.forEachIndexed { idx, e ->
            if (e.prevSha256 != expectedPrev) {
                // Walk forward through audited deletions to see if they explain the gap.
                var cur = expectedPrev
                var hops = 0
                while (hops < entries.size + bridges.size && cur != e.prevSha256) {
                    cur = bridges[cur] ?: break
                    hops++
                }
                if (cur != e.prevSha256) {
                    return@withContext Status.Broken(
                        position = idx + 1,
                        total = entries.size,
                        entryId = e.id,
                        reason = "link mismatch with no audited deletion to explain it"
                    )
                }
                bridged += hops
            }

            if (!StorageLocationManager.exists(ctx, e.filePath)) {
                audioAbsent++
            } else if (!contentSet.contains(e.id)) {
                notChecked++
            } else {
                val stored = StorageLocationManager.readSegment(ctx, e.filePath)
                if (stored == null) {
                    // Vanished between the exists() check and the read.
                    audioAbsent++
                } else {
                    val plain = runCatching {
                        KeystoreManager.decrypt(stored, e.classification == "PROTECTED")
                    }.getOrNull()

                    if (plain == null) {
                        return@withContext Status.Broken(
                            position = idx + 1,
                            total = entries.size,
                            entryId = e.id,
                            reason = "decryption failed - ciphertext or GCM tag altered"
                        )
                    }
                    if (!HashChainManager.verifyEntryHash(plain, e.prevSha256, e.sha256)) {
                        return@withContext Status.Broken(
                            position = idx + 1,
                            total = entries.size,
                            entryId = e.id,
                            reason = "content hash mismatch - audio differs from what was recorded"
                        )
                    }
                    contentVerified++
                }
            }
            expectedPrev = e.sha256
        }

        Status.Verified(entries.size, contentVerified, audioAbsent, notChecked, bridged)
    }
}
