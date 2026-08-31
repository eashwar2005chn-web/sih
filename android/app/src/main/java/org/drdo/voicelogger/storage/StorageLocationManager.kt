package org.drdo.voicelogger.storage

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import androidx.documentfile.provider.DocumentFile
import org.drdo.voicelogger.data.prefs.AppSettings
import java.io.File

/**
 * Resolves where encrypted logbook segments live, and reads/writes/deletes them there.
 *
 * Two backing stores are supported and a single [LogbookEntry.filePath] string identifies which:
 *
 *   - app-private internal storage  -> an absolute path  (/data/user/0/<pkg>/files/segment_x.bin)
 *   - a user-chosen SAF tree        -> a content:// URI  (persisted grant, survives reboot)
 *
 * Callers never branch on the backing store; they pass the stored location string back in.
 * That keeps the Room schema unchanged (no migration) while letting the operator move the
 * corpus onto an SD card or a shared folder.
 *
 * IMPORTANT: a chosen directory can stop being writable between runs - the card is removed,
 * the folder is deleted, or the grant is revoked from Android settings. Every write therefore
 * re-checks the grant and falls back to internal storage rather than dropping audio on the
 * floor. Losing a recording is a far worse failure than writing it to the wrong place.
 */
object StorageLocationManager {

    private const val TAG = "StorageLocation"
    private const val MIME = "application/octet-stream"

    /** True when [location] refers to the SAF tree rather than a plain file path. */
    private fun isContentUri(location: String) = location.startsWith("content://")

    /**
     * The active tree URI if one is set AND we still hold a persisted write grant for it.
     * Returns null (meaning "use internal storage") in every other case.
     */
    fun activeTreeUri(ctx: Context): Uri? {
        val raw = AppSettings.getStorageTreeUri(ctx) ?: return null
        val uri = runCatching { Uri.parse(raw) }.getOrNull() ?: return null
        val held = ctx.contentResolver.persistedUriPermissions.any {
            it.uri == uri && it.isWritePermission
        }
        if (!held) {
            android.util.Log.w(TAG, "Persisted write grant for $uri is gone; using internal storage")
            return null
        }
        return uri
    }

    fun isUsingExternal(ctx: Context): Boolean = activeTreeUri(ctx) != null

    /** Human-readable description of the current destination, for the settings screen. */
    fun describe(ctx: Context): String {
        val uri = activeTreeUri(ctx)
            ?: return "App-private internal storage (default)"
        val name = runCatching { DocumentFile.fromTreeUri(ctx, uri)?.name }.getOrNull()
        return name ?: uri.lastPathSegment ?: uri.toString()
    }

    /**
     * Writes [bytes] as [name] and returns the location string to persist on the entry.
     * Falls back to internal storage if the SAF write fails for any reason.
     */
    fun writeSegment(ctx: Context, name: String, bytes: ByteArray): String {
        val tree = activeTreeUri(ctx)
        if (tree != null) {
            var created: DocumentFile? = null
            val written = runCatching {
                val dir = DocumentFile.fromTreeUri(ctx, tree)
                    ?: error("tree URI did not resolve to a document")
                val doc = dir.createFile(MIME, name)
                    ?: error("createFile returned null for $name")
                created = doc
                ctx.contentResolver.openOutputStream(doc.uri)?.use { it.write(bytes) }
                    ?: error("openOutputStream returned null")
                doc.uri.toString()
            }.getOrElse { e ->
                android.util.Log.e(TAG, "SAF write failed (${e.message}); falling back to internal")
                // createFile may have succeeded before the stream failed. Remove the empty
                // document rather than leaving a 0-byte segment in the folder that no logbook
                // row points at and nothing in the app can clean up.
                runCatching { created?.delete() }
                null
            }
            if (written != null) return written
        }
        val f = File(ctx.filesDir, name)
        f.outputStream().use { it.write(bytes) }
        return f.absolutePath
    }

    /** Reads a segment back, or null if it is gone. */
    fun readSegment(ctx: Context, location: String): ByteArray? = runCatching {
        if (isContentUri(location)) {
            ctx.contentResolver.openInputStream(Uri.parse(location))?.use { it.readBytes() }
        } else {
            File(location).takeIf { it.exists() }?.readBytes()
        }
    }.getOrNull()

    fun exists(ctx: Context, location: String): Boolean =
        if (isContentUri(location)) {
            runCatching {
                DocumentFile.fromSingleUri(ctx, Uri.parse(location))?.exists() == true
            }.getOrDefault(false)
        } else {
            File(location).exists()
        }

    fun sizeBytes(ctx: Context, location: String): Long =
        if (isContentUri(location)) {
            runCatching {
                DocumentFile.fromSingleUri(ctx, Uri.parse(location))?.length() ?: 0L
            }.getOrDefault(0L)
        } else {
            File(location).let { if (it.exists()) it.length() else 0L }
        }

    /**
     * Deletes the backing file. Returns true if the file is gone afterwards - including when it
     * was already missing, since the caller's goal is absence, not the act of deleting.
     */
    fun deleteSegment(ctx: Context, location: String): Boolean = runCatching {
        if (isContentUri(location)) {
            val uri = Uri.parse(location)
            DocumentsContract.deleteDocument(ctx.contentResolver, uri)
            !exists(ctx, location)
        } else {
            val f = File(location)
            if (!f.exists()) true else f.delete()
        }
    }.getOrElse { e ->
        android.util.Log.e(TAG, "delete failed for $location: ${e.message}")
        !exists(ctx, location)
    }
}
