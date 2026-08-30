package org.drdo.voicelogger.crypto

import java.security.MessageDigest

/**
 * Logbook Hash Chain Engine.
 * Ensures tamper-evident auditability by chaining each logbook segment's SHA-256 hash
 * with the preceding entry's hash.
 */
object HashChainManager {
    const val GENESIS_PREV_HASH = "GENESIS_LOGBOOK_SEED_DRDO_PS26052"

    /**
     * Computes the SHA-256 hash for a logbook entry's audio payload linked to prevSha256.
     * Returns a 64-character hexadecimal SHA-256 string.
     */
    fun computeEntryHash(audioBytes: ByteArray, prevSha256: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(audioBytes)
        digest.update(prevSha256.toByteArray(Charsets.UTF_8))
        val hashBytes = digest.digest()
        return bytesToHex(hashBytes)
    }

    /**
     * Validates whether an entry's hash matches its audio content and prevSha256 link.
     */
    fun verifyEntryHash(audioBytes: ByteArray, prevSha256: String, expectedSha256: String): Boolean {
        val calculated = computeEntryHash(audioBytes, prevSha256)
        return calculated.equals(expectedSha256, ignoreCase = true)
    }

    private fun bytesToHex(bytes: ByteArray): String {
        val hexChars = CharArray(bytes.size * 2)
        val hexArray = "0123456789abcdef".toCharArray()
        for (i in bytes.indices) {
            val v = bytes[i].toInt() and 0xFF
            hexChars[i * 2] = hexArray[v ushr 4]
            hexChars[i * 2 + 1] = hexArray[v and 0x0F]
        }
        return String(hexChars)
    }
}
