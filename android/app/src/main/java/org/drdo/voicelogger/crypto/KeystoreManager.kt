package org.drdo.voicelogger.crypto

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Manages AES-256-GCM encryption at rest using the Android Keystore.
 * Provides distinct key aliases for normal logbook entries vs. top-secret duress entries.
 */
object KeystoreManager {
    private const val ANDROID_KEYSTORE = "AndroidKeyStore"
    private const val ALGORITHM = KeyProperties.KEY_ALGORITHM_AES
    private const val BLOCK_MODE = KeyProperties.BLOCK_MODE_GCM
    private const val PADDING = KeyProperties.ENCRYPTION_PADDING_NONE
    private const val TRANSFORMATION = "$ALGORITHM/$BLOCK_MODE/$PADDING"
    private const val GCM_TAG_LENGTH = 128

    const val PRIMARY_KEY_ALIAS = "VoiceLoggerPrimaryKey"
    const val PROTECTED_KEY_ALIAS = "VoiceLoggerProtectedKey"

    init {
        getOrCreateSecretKey(PRIMARY_KEY_ALIAS)
        getOrCreateSecretKey(PROTECTED_KEY_ALIAS)
    }

    @Synchronized
    private fun getOrCreateSecretKey(alias: String): SecretKey {
        val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
        if (keyStore.containsAlias(alias)) {
            val entry = keyStore.getEntry(alias, null) as KeyStore.SecretKeyEntry
            return entry.secretKey
        }

        val keyGenerator = KeyGenerator.getInstance(ALGORITHM, ANDROID_KEYSTORE)
        val spec = KeyGenParameterSpec.Builder(
            alias,
            KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
        )
            .setBlockModes(BLOCK_MODE)
            .setEncryptionPaddings(PADDING)
            .setKeySize(256)
            .build()

        keyGenerator.init(spec)
        return keyGenerator.generateKey()
    }

    /**
     * Encrypts plaintext bytes using AES-256-GCM with the specified Keystore alias.
     * Returns ByteArray containing IV (12 bytes) + Ciphertext.
     */
    fun encrypt(data: ByteArray, isProtected: Boolean = false): ByteArray {
        val alias = if (isProtected) PROTECTED_KEY_ALIAS else PRIMARY_KEY_ALIAS
        val secretKey = getOrCreateSecretKey(alias)
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, secretKey)

        val iv = cipher.iv
        val ciphertext = cipher.doFinal(data)

        // Pack IV (12 bytes) + Ciphertext
        val combined = ByteArray(iv.size + ciphertext.size)
        System.arraycopy(iv, 0, combined, 0, iv.size)
        System.arraycopy(ciphertext, 0, combined, iv.size, ciphertext.size)
        return combined
    }

    /**
     * Decrypts encrypted bytes using AES-256-GCM with the specified Keystore alias.
     */
    fun decrypt(encryptedCombined: ByteArray, isProtected: Boolean = false): ByteArray {
        val alias = if (isProtected) PROTECTED_KEY_ALIAS else PRIMARY_KEY_ALIAS
        val secretKey = getOrCreateSecretKey(alias)

        val ivSize = 12
        val iv = ByteArray(ivSize)
        val ciphertext = ByteArray(encryptedCombined.size - ivSize)

        System.arraycopy(encryptedCombined, 0, iv, 0, ivSize)
        System.arraycopy(encryptedCombined, ivSize, ciphertext, 0, ciphertext.size)

        val cipher = Cipher.getInstance(TRANSFORMATION)
        val spec = GCMParameterSpec(GCM_TAG_LENGTH, iv)
        cipher.init(Cipher.DECRYPT_MODE, secretKey, spec)

        return cipher.doFinal(ciphertext)
    }
}
