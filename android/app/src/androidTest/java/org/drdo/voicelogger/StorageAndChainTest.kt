package org.drdo.voicelogger

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.drdo.voicelogger.crypto.HashChainManager
import org.drdo.voicelogger.crypto.KeystoreManager
import org.drdo.voicelogger.data.prefs.AppSettings
import org.drdo.voicelogger.storage.StorageLocationManager
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.UUID

/**
 * Covers the paths added when storage became configurable and the chain claim became real.
 *
 * These tests deliberately do NOT touch the shared logbook database - on a device that has
 * been recording, that database holds the operator's actual recordings, and a test that
 * inserts or purges rows there would destroy real evidence to check its own assertions.
 * Everything here works on uniquely-named scratch files instead.
 */
@RunWith(AndroidJUnit4::class)
class StorageAndChainTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()

    private fun scratchName() = "test_segment_" + UUID.randomUUID() + ".bin"

    @Test
    fun internalStorageRoundTrip() {
        val name = scratchName()
        val payload = ByteArray(4096) { (it % 251).toByte() }

        val location = StorageLocationManager.writeSegment(context, name, payload)
        try {
            assertTrue("A location string must be returned", location.isNotEmpty())
            assertTrue("File must exist after write", StorageLocationManager.exists(context, location))
            assertEquals(
                "Reported size must match what was written",
                payload.size.toLong(),
                StorageLocationManager.sizeBytes(context, location)
            )

            val read = StorageLocationManager.readSegment(context, location)
            assertNotNull("Segment must be readable back", read)
            assertTrue("Bytes must round-trip unchanged", payload.contentEquals(read!!))
        } finally {
            assertTrue(
                "Delete must report success",
                StorageLocationManager.deleteSegment(context, location)
            )
        }
        assertFalse("File must be gone after delete", StorageLocationManager.exists(context, location))
        assertNull(
            "Reading a deleted segment must return null, not stale bytes",
            StorageLocationManager.readSegment(context, location)
        )
    }

    @Test
    fun deletingAnAbsentSegmentIsNotAnError() {
        // The caller wants the file gone; a file that was already gone satisfies that.
        val phantom = context.filesDir.absolutePath + "/never_written_" + UUID.randomUUID() + ".bin"
        assertTrue(StorageLocationManager.deleteSegment(context, phantom))
    }

    @Test
    fun defaultDestinationIsInternalWhenNoTreeIsSet() {
        val saved = AppSettings.getStorageTreeUri(context)
        try {
            AppSettings.setStorageTreeUri(context, null)
            assertFalse(StorageLocationManager.isUsingExternal(context))
            assertNull(StorageLocationManager.activeTreeUri(context))
        } finally {
            AppSettings.setStorageTreeUri(context, saved)
        }
    }

    @Test
    fun revokedTreeGrantFallsBackToInternal() {
        val saved = AppSettings.getStorageTreeUri(context)
        try {
            // A tree URI we hold no persisted grant for - exactly the state after the operator
            // revokes folder access, or the SD card is removed. Capture must not stop.
            AppSettings.setStorageTreeUri(
                context, "content://com.android.externalstorage.documents/tree/primary%3ANotGranted"
            )
            assertNull(
                "An ungranted tree must resolve to null so writes fall back",
                StorageLocationManager.activeTreeUri(context)
            )

            val name = scratchName()
            val location = StorageLocationManager.writeSegment(context, name, ByteArray(64) { 7 })
            assertTrue(
                "Fallback must write to internal storage, not a content URI",
                location.startsWith(context.filesDir.absolutePath)
            )
            StorageLocationManager.deleteSegment(context, location)
        } finally {
            AppSettings.setStorageTreeUri(context, saved)
        }
    }

    /**
     * Reproduces exactly what the service does per segment - encrypt, store, then hash the
     * PLAINTEXT - and confirms a verifier reading the file back reaches the same conclusion.
     * If these two ever diverge, every entry would verify as broken.
     */
    @Test
    fun storedSegmentVerifiesAgainstItsChainHash() {
        val plain = ByteArray(8192) { ((it * 31) % 256).toByte() }
        val prev = HashChainManager.GENESIS_PREV_HASH
        val sha = HashChainManager.computeEntryHash(plain, prev)

        val location = StorageLocationManager.writeSegment(
            context, scratchName(), KeystoreManager.encrypt(plain, false)
        )
        try {
            val stored = StorageLocationManager.readSegment(context, location)
            assertNotNull(stored)
            val decrypted = KeystoreManager.decrypt(stored!!, false)
            assertTrue("Decrypted bytes must match the original", plain.contentEquals(decrypted))
            assertTrue(
                "Hash recomputed from disk must match the stored chain hash",
                HashChainManager.verifyEntryHash(decrypted, prev, sha)
            )
        } finally {
            StorageLocationManager.deleteSegment(context, location)
        }
    }

    /** GCM must reject a modified ciphertext rather than returning corrupt plaintext. */
    @Test
    fun tamperedCiphertextFailsToDecrypt() {
        val plain = "SEGMENT_UNDER_TEST".toByteArray()
        val encrypted = KeystoreManager.encrypt(plain, false)
        // Flip a byte in the ciphertext body, past the 12-byte IV.
        encrypted[encrypted.size - 5] = (encrypted[encrypted.size - 5] + 1).toByte()

        val result = runCatching { KeystoreManager.decrypt(encrypted, false) }
        assertTrue("AES-GCM must reject a tampered payload", result.isFailure)
    }

    @Test
    fun protectedAndNormalEntriesUseDistinctKeys() {
        val plain = "DURESS_PAYLOAD".toByteArray()
        val protectedBlob = KeystoreManager.encrypt(plain, isProtected = true)

        // Decrypting a protected segment with the normal key must fail: the top-secret store is
        // only meaningfully separate if the keys actually differ.
        val wrongKey = runCatching { KeystoreManager.decrypt(protectedBlob, isProtected = false) }
        assertTrue("Protected payload must not decrypt under the primary key", wrongKey.isFailure)

        val rightKey = KeystoreManager.decrypt(protectedBlob, isProtected = true)
        assertTrue(plain.contentEquals(rightKey))
    }
}
