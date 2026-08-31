package org.drdo.voicelogger.audio

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.drdo.voicelogger.crypto.KeystoreManager
import org.drdo.voicelogger.data.model.LogbookEntry
import org.drdo.voicelogger.storage.StorageLocationManager
import java.io.ByteArrayOutputStream

/**
 * Plays back a stored segment: read -> decrypt -> AudioTrack.
 *
 * Segments are raw 16 kHz mono PCM-16 encrypted with AES-256-GCM, so there is no container to
 * hand to MediaPlayer; the bytes are streamed to an AudioTrack directly. Only one segment
 * plays at a time - starting a second stops the first.
 *
 * Playback is a read path and deliberately touches nothing else. It never re-writes the file,
 * so listening to a recording cannot change its hash.
 */
class SegmentPlayer(private val ctx: Context) {

    companion object {
        const val SAMPLE_RATE = 16000
    }

    interface Listener {
        fun onStarted(entryId: String)
        fun onFinished(entryId: String)
        fun onError(entryId: String, message: String)
    }

    private var track: AudioTrack? = null
    private var job: Job? = null
    private val scope = CoroutineScope(Dispatchers.IO)

    @Volatile var playingId: String? = null
        private set

    var listener: Listener? = null

    fun isPlaying(id: String) = playingId == id

    fun toggle(entry: LogbookEntry) {
        if (playingId == entry.id) stop() else play(entry)
    }

    fun play(entry: LogbookEntry) {
        stop()
        playingId = entry.id
        job = scope.launch {
            val pcm = decodeToPcm(entry)
            if (pcm == null) {
                playingId = null
                withContext(Dispatchers.Main) {
                    listener?.onError(entry.id, "Audio missing or could not be decrypted")
                }
                return@launch
            }
            withContext(Dispatchers.Main) { listener?.onStarted(entry.id) }

            val minBuf = AudioTrack.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT
            ).coerceAtLeast(4096)

            val t = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setBufferSizeInBytes(minBuf)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()

            track = t
            t.play()
            var off = 0
            while (off < pcm.size && playingId == entry.id) {
                val n = t.write(pcm, off, minOf(minBuf, pcm.size - off))
                if (n <= 0) break
                off += n
            }
            runCatching { t.stop(); t.release() }
            track = null
            val finishedNaturally = playingId == entry.id
            playingId = null
            if (finishedNaturally) {
                withContext(Dispatchers.Main) { listener?.onFinished(entry.id) }
            }
        }
    }

    fun stop() {
        val wasPlaying = playingId
        playingId = null
        job?.cancel()
        job = null
        runCatching { track?.pause(); track?.flush(); track?.stop(); track?.release() }
        track = null
        if (wasPlaying != null) listener?.onFinished(wasPlaying)
    }

    fun release() {
        stop()
    }

    private fun decodeToPcm(entry: LogbookEntry): ByteArray? {
        val stored = StorageLocationManager.readSegment(ctx, entry.filePath) ?: return null
        return runCatching {
            KeystoreManager.decrypt(stored, entry.classification == "PROTECTED")
        }.getOrNull()
    }

    /** Decrypted PCM wrapped in a 44-byte WAV header, for export/share. Null if unavailable. */
    fun exportWav(entry: LogbookEntry): ByteArray? {
        val pcm = decodeToPcm(entry) ?: return null
        return wavWrap(pcm, SAMPLE_RATE, 1, 16)
    }

    private fun wavWrap(pcm: ByteArray, rate: Int, channels: Int, bits: Int): ByteArray {
        val byteRate = rate * channels * bits / 8
        val out = ByteArrayOutputStream(44 + pcm.size)
        fun s(v: String) = out.write(v.toByteArray(Charsets.US_ASCII))
        fun i32(v: Int) = out.write(
            byteArrayOf(
                (v and 0xFF).toByte(), ((v shr 8) and 0xFF).toByte(),
                ((v shr 16) and 0xFF).toByte(), ((v shr 24) and 0xFF).toByte()
            )
        )
        fun i16(v: Int) = out.write(byteArrayOf((v and 0xFF).toByte(), ((v shr 8) and 0xFF).toByte()))

        s("RIFF"); i32(36 + pcm.size); s("WAVE")
        s("fmt "); i32(16); i16(1); i16(channels); i32(rate); i32(byteRate)
        i16(channels * bits / 8); i16(bits)
        s("data"); i32(pcm.size)
        out.write(pcm)
        return out.toByteArray()
    }
}
