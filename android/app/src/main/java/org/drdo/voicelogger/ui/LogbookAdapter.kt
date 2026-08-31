package org.drdo.voicelogger.ui

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import org.drdo.voicelogger.R
import org.drdo.voicelogger.data.model.LogbookEntry
import org.drdo.voicelogger.databinding.ItemLogbookBinding

/**
 * Logbook list.
 *
 * ListAdapter + DiffUtil rather than notifyDataSetChanged: entries arrive one at a time from a
 * Room Flow, and diffing is what makes a new row animate in instead of the whole list blinking.
 */
class LogbookAdapter(
    private val onPlay: (LogbookEntry) -> Unit,
    private val onShare: (LogbookEntry) -> Unit,
    private val onDelete: (LogbookEntry) -> Unit
) : ListAdapter<LogbookEntry, LogbookAdapter.VH>(DIFF) {

    companion object {
        private val DIFF = object : DiffUtil.ItemCallback<LogbookEntry>() {
            override fun areItemsTheSame(a: LogbookEntry, b: LogbookEntry) = a.id == b.id
            override fun areContentsTheSame(a: LogbookEntry, b: LogbookEntry) =
                a.syncState == b.syncState && a.sizeBytes == b.sizeBytes &&
                        a.classification == b.classification && a.sha256 == b.sha256
        }
    }

    /** Id of the entry currently playing, so exactly one row shows a stop icon. */
    var playingId: String? = null
        set(value) {
            val old = field
            field = value
            currentList.forEachIndexed { i, e ->
                if (e.id == old || e.id == value) notifyItemChanged(i)
            }
        }

    class VH(val v: ItemLogbookBinding) : RecyclerView.ViewHolder(v.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH =
        VH(ItemLogbookBinding.inflate(LayoutInflater.from(parent.context), parent, false))

    override fun onBindViewHolder(h: VH, position: Int) {
        val e = getItem(position)
        val ctx = h.v.root.context

        h.v.txtTime.text = Formats.timestamp(e.startedAt)

        val lr = e.levelReductionDb
        val lrText = if (lr != null)
            String.format(java.util.Locale.UK, "%.2f dB reduction", lr)
        else
            "level not measured"
        h.v.txtMeta.text = Formats.duration(e.durationSec) + "  ·  " +
                Formats.bytes(e.sizeBytes) + "  ·  " + lrText

        h.v.txtHash.text = "sha256 " + e.sha256.take(16) + "..."

        // Classification badge
        val badgeColour = when (e.classification) {
            "PROTECTED" -> R.color.warn
            "SOS" -> R.color.danger
            else -> R.color.blue_600
        }
        h.v.badgeClass.text = e.classification
        h.v.badgeClass.setTextColor(ContextCompat.getColor(ctx, badgeColour))

        // Sync state
        val syncColour = when (e.syncState) {
            "synced" -> R.color.success
            "failed" -> R.color.danger
            "uploading" -> R.color.blue_600
            else -> R.color.text_muted
        }
        h.v.txtSyncBadge.text = if (e.syncState == "failed" && e.lastError != null)
            "failed: " + e.lastError else e.syncState
        h.v.txtSyncBadge.setTextColor(ContextCompat.getColor(ctx, syncColour))

        val isPlaying = e.id == playingId
        h.v.btnPlay.setIconResource(if (isPlaying) R.drawable.ic_stop else R.drawable.ic_play)

        h.v.btnPlay.setOnClickListener { onPlay(e) }
        h.v.btnShare.setOnClickListener { onShare(e) }
        h.v.btnDelete.setOnClickListener { onDelete(e) }
    }
}
