package org.drdo.voicelogger.ui.widget

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import org.drdo.voicelogger.R

/**
 * Horizontal input-level meter with peak hold.
 *
 * Fed real dBFS from the capture loop. Two things make it read smoothly without lying about
 * the signal:
 *
 *   BALLISTICS  The bar rises almost instantly (fast attack) and falls gradually (slow
 *               release), the same asymmetry a hardware VU meter uses. A meter that tracked
 *               the raw per-block RMS would flicker illegibly on speech.
 *   PEAK HOLD   A thin marker sits at the loudest level seen in the last moment and decays.
 *               This is what actually tells the operator whether the input is clipping.
 *
 * Smoothing changes how the value is DRAWN, never what is recorded or reported: it is applied
 * here in the view, downstream of everything that touches the audio.
 */
class LevelMeterView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0
) : View(context, attrs, defStyle) {

    companion object {
        private const val FLOOR_DB = -60f
        private const val ATTACK = 0.55f     // fraction of the gap closed per frame when rising
        private const val RELEASE = 0.10f    // ... and when falling
        private const val PEAK_DECAY = 0.012f
    }

    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.border)
    }
    private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.blue_600)
    }
    private val peakPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.blue_500)
    }
    private val hotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.danger)
    }

    private val rect = RectF()

    /** Target fill in 0..1, derived from the last dBFS value handed in. */
    private var target = 0f
    private var shown = 0f
    private var peak = 0f
    private var animating = false

    /** Feed a measured level. -60 dBFS or quieter reads as empty. */
    fun setLevelDbfs(db: Float) {
        target = ((db - FLOOR_DB) / -FLOOR_DB).coerceIn(0f, 1f)
        if (!animating) {
            animating = true
            postInvalidateOnAnimation()
        }
    }

    fun reset() {
        target = 0f
        shown = 0f
        peak = 0f
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val gap = target - shown
        shown += gap * (if (gap > 0) ATTACK else RELEASE)
        if (shown > peak) peak = shown else peak = (peak - PEAK_DECAY).coerceAtLeast(shown)

        val h = height.toFloat()
        val w = width.toFloat()
        val r = h / 2f

        rect.set(0f, 0f, w, h)
        canvas.drawRoundRect(rect, r, r, trackPaint)

        if (shown > 0.001f) {
            val fill = (w * shown).coerceAtLeast(h)
            rect.set(0f, 0f, fill, h)
            // Above -6 dBFS the input is close to clipping; the bar says so in red.
            canvas.drawRoundRect(rect, r, r, if (shown > 0.9f) hotPaint else barPaint)
        }

        if (peak > 0.01f) {
            val x = (w * peak).coerceIn(h, w - 3f)
            rect.set(x - 3f, 0f, x, h)
            canvas.drawRoundRect(rect, 2f, 2f, if (peak > 0.9f) hotPaint else peakPaint)
        }

        // Keep animating while there is still motion left to render, then stop cleanly so an
        // idle meter costs nothing.
        if (kotlin.math.abs(target - shown) > 0.002f || peak > shown + 0.002f) {
            postInvalidateOnAnimation()
        } else {
            animating = false
        }
    }
}
