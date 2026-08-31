package org.drdo.voicelogger.ui

import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/** Display formatting shared across the screens. Presentation only - no value is rounded before it is stored. */
object Formats {

    private val dayFmt: DateTimeFormatter =
        DateTimeFormatter.ofPattern("dd MMM  HH:mm:ss", Locale.UK).withZone(ZoneId.systemDefault())

    fun timestamp(iso: String): String = runCatching {
        dayFmt.format(Instant.parse(iso))
    }.getOrDefault(iso)

    fun bytes(b: Long): String = when {
        b <= 0L -> "0 B"
        b < 1024L -> b.toString() + " B"
        b < 1024L * 1024 -> String.format(Locale.UK, "%.1f KB", b / 1024.0)
        b < 1024L * 1024 * 1024 -> String.format(Locale.UK, "%.1f MB", b / 1048576.0)
        else -> String.format(Locale.UK, "%.2f GB", b / 1073741824.0)
    }

    /** h:mm:ss for anything an hour or longer, m:ss below that. */
    fun duration(totalSec: Double): String {
        val s = totalSec.toLong().coerceAtLeast(0L)
        val h = s / 3600
        val m = (s % 3600) / 60
        val sec = s % 60
        return if (h > 0) String.format(Locale.UK, "%d:%02d:%02d", h, m, sec)
        else String.format(Locale.UK, "%d:%02d", m, sec)
    }
}
