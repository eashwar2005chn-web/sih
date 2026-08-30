package org.drdo.voicelogger.anc

/**
 * Self-contained 512-point Cooley-Tukey Radix-2 Fast Fourier Transform (FFT) & Inverse FFT (IFFT).
 */
object FFT512 {
    private const val N = 512
    private val cosTable = FloatArray(N / 2)
    private val sinTable = FloatArray(N / 2)
    private val bitRev = IntArray(N)

    init {
        for (i in 0 until N / 2) {
            cosTable[i] = kotlin.math.cos(2.0 * Math.PI * i / N).toFloat()
            sinTable[i] = kotlin.math.sin(2.0 * Math.PI * i / N).toFloat()
        }
        for (i in 0 until N) {
            var rev = 0
            var temp = i
            for (j in 0 until 9) { // log2(512) = 9
                rev = (rev shl 1) or (temp and 1)
                temp = temp ushr 1
            }
            bitRev[i] = rev
        }
    }

    fun fft(realIn: FloatArray, imagIn: FloatArray, realOut: FloatArray, imagOut: FloatArray) {
        transform(realIn, imagIn, realOut, imagOut, isInverse = false)
    }

    fun ifft(realIn: FloatArray, imagIn: FloatArray, realOut: FloatArray, imagOut: FloatArray) {
        transform(realIn, imagIn, realOut, imagOut, isInverse = true)
        for (i in 0 until N) {
            realOut[i] /= N.toFloat()
            imagOut[i] /= N.toFloat()
        }
    }

    private fun transform(
        realIn: FloatArray,
        imagIn: FloatArray,
        realOut: FloatArray,
        imagOut: FloatArray,
        isInverse: Boolean
    ) {
        for (i in 0 until N) {
            val rev = bitRev[i]
            realOut[i] = realIn[rev]
            imagOut[i] = imagIn[rev]
        }

        var len = 2
        while (len <= N) {
            val halfLen = len / 2
            val step = N / len
            var i = 0
            while (i < N) {
                var k = 0
                for (j in 0 until halfLen) {
                    val cosW = cosTable[k]
                    val sinW = if (isInverse) -sinTable[k] else sinTable[k]

                    val uR = realOut[i + j]
                    val uI = imagOut[i + j]

                    val vR = realOut[i + j + halfLen] * cosW - imagOut[i + j + halfLen] * sinW
                    val vI = realOut[i + j + halfLen] * sinW + imagOut[i + j + halfLen] * cosW

                    realOut[i + j] = uR + vR
                    imagOut[i + j] = uI + vI

                    realOut[i + j + halfLen] = uR - vR
                    imagOut[i + j + halfLen] = uI - vI

                    k += step
                }
                i += len
            }
            len = len shl 1
        }
    }
}
