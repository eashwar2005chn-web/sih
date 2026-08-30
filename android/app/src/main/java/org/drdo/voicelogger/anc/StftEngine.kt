package org.drdo.voicelogger.anc

/**
 * High-precision Short-Time Fourier Transform (STFT) & Inverse STFT (iSTFT) Engine.
 * Specifications:
 * - n_fft = 512
 * - hop_length = 256
 * - win_length = 512
 * - window = Hann window
 * - center = true (reflect-padding by 256 samples)
 * - Frequency Bins = 257 (0..256)
 */
class StftEngine {
    val nFft = 512
    val hopLength = 256
    val winLength = 512
    val nFreq = 257

    private val window = FloatArray(winLength)

    init {
        for (i in 0 until winLength) {
            window[i] = 0.5f * (1.0f - kotlin.math.cos(2.0 * Math.PI * i / winLength).toFloat())
        }
    }

    data class StftResult(
        val real: Array<FloatArray>, // Shape: (257, T)
        val imag: Array<FloatArray>, // Shape: (257, T)
        val timeFrames: Int
    )

    fun stft(pcm: FloatArray): StftResult {
        val origLen = pcm.size
        val padAmount = nFft / 2 // 256

        // 1. Reflect-pad PCM by 256 samples at start and end
        val paddedLen = origLen + 2 * padAmount
        val padded = FloatArray(paddedLen)

        // Center copy
        System.arraycopy(pcm, 0, padded, padAmount, origLen)

        // Left reflect-pad
        for (i in 0 until padAmount) {
            val srcIdx = padAmount - i
            padded[i] = if (srcIdx < origLen) pcm[srcIdx] else 0.0f
        }

        // Right reflect-pad
        for (i in 0 until padAmount) {
            val srcIdx = origLen - 2 - i
            padded[origLen + padAmount + i] = if (srcIdx >= 0) pcm[srcIdx] else 0.0f
        }

        val timeFrames = (paddedLen - nFft) / hopLength + 1
        val realMat = Array(nFreq) { FloatArray(timeFrames) }
        val imagMat = Array(nFreq) { FloatArray(timeFrames) }

        val rIn = FloatArray(nFft)
        val iIn = FloatArray(nFft)
        val rOut = FloatArray(nFft)
        val iOut = FloatArray(nFft)

        for (t in 0 until timeFrames) {
            val startIdx = t * hopLength
            for (i in 0 until nFft) {
                rIn[i] = padded[startIdx + i] * window[i]
                iIn[i] = 0.0f
            }

            FFT512.fft(rIn, iIn, rOut, iOut)

            for (f in 0 until nFreq) {
                realMat[f][t] = rOut[f]
                imagMat[f][t] = iOut[f]
            }
        }

        return StftResult(realMat, imagMat, timeFrames)
    }

    fun istft(realMat: Array<FloatArray>, imagMat: Array<FloatArray>, originalLength: Int): FloatArray {
        val timeFrames = realMat[0].size
        val paddedLen = (timeFrames - 1) * hopLength + nFft

        val outBuf = FloatArray(paddedLen)
        val winBuf = FloatArray(paddedLen)

        val rIn = FloatArray(nFft)
        val iIn = FloatArray(nFft)
        val rOut = FloatArray(nFft)
        val iOut = FloatArray(nFft)

        for (t in 0 until timeFrames) {
            val startIdx = t * hopLength

            // Construct 512-point spectrum with conjugate symmetry
            for (f in 0 until nFreq) {
                rIn[f] = realMat[f][t]
                iIn[f] = imagMat[f][t]
            }
            for (f in 1 until nFreq - 1) {
                val mirrorIdx = nFft - f
                rIn[mirrorIdx] = realMat[f][t]
                iIn[mirrorIdx] = -imagMat[f][t]
            }

            FFT512.ifft(rIn, iIn, rOut, iOut)

            for (i in 0 until nFft) {
                val pos = startIdx + i
                if (pos < paddedLen) {
                    outBuf[pos] += rOut[i] * window[i]
                    winBuf[pos] += window[i] * window[i]
                }
            }
        }

        // Window energy normalization & reflect-unpad
        val result = FloatArray(originalLength)
        val padAmount = nFft / 2 // 256

        for (i in 0 until originalLength) {
            val pos = padAmount + i
            if (pos < paddedLen && winBuf[pos] > 1e-4f) {
                result[i] = outBuf[pos] / winBuf[pos]
            } else if (pos < paddedLen) {
                result[i] = outBuf[pos]
            }
        }

        return result
    }
}
