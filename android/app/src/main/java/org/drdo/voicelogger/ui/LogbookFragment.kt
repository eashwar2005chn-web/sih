package org.drdo.voicelogger.ui

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.DefaultItemAnimator
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.drdo.voicelogger.R
import org.drdo.voicelogger.audio.SegmentPlayer
import org.drdo.voicelogger.crypto.ChainVerifier
import org.drdo.voicelogger.data.db.AppDatabase
import org.drdo.voicelogger.data.model.LogbookEntry
import org.drdo.voicelogger.data.repo.LogbookRepository
import org.drdo.voicelogger.databinding.FragmentLogbookBinding
import java.io.File

/**
 * The logbook: every recorded segment, with playback, export and deletion.
 *
 * The chain card at the top runs a real verification pass. It has three distinct outcomes -
 * nothing recorded yet, verified, and broken at a specific entry - and it never claims the
 * third-party-checkable word "verified" without having checked.
 */
class LogbookFragment : Fragment() {

    private var _b: FragmentLogbookBinding? = null
    private val b get() = _b!!

    private lateinit var adapter: LogbookAdapter
    private var player: SegmentPlayer? = null

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, saved: Bundle?
    ): View {
        _b = FragmentLogbookBinding.inflate(inflater, container, false)
        return b.root
    }

    override fun onViewCreated(view: View, saved: Bundle?) {
        super.onViewCreated(view, saved)

        player = SegmentPlayer(requireContext().applicationContext).also { p ->
            p.listener = object : SegmentPlayer.Listener {
                override fun onStarted(entryId: String) {
                    adapter.playingId = entryId
                }

                override fun onFinished(entryId: String) {
                    if (adapter.playingId == entryId) adapter.playingId = null
                }

                override fun onError(entryId: String, message: String) {
                    adapter.playingId = null
                    _b?.let { Snackbar.make(it.root, message, Snackbar.LENGTH_LONG).show() }
                }
            }
        }

        adapter = LogbookAdapter(
            onPlay = { entry -> player?.toggle(entry) },
            onShare = { entry -> exportAndShare(entry) },
            onDelete = { entry -> confirmDeleteOne(entry) }
        )

        b.recyclerLogbook.layoutManager = LinearLayoutManager(requireContext())
        b.recyclerLogbook.adapter = adapter
        b.recyclerLogbook.itemAnimator = DefaultItemAnimator().apply {
            addDuration = 180
            removeDuration = 160
            moveDuration = 180
        }

        b.btnVerifyChain.setOnClickListener { runVerification() }
        b.btnDeleteAll.setOnClickListener { confirmDeleteAll() }

        viewLifecycleOwner.lifecycleScope.launch {
            AppDatabase.getDatabase(requireContext()).logbookEntryDao()
                .getAllEntriesFlow().collectLatest { list ->
                    adapter.submitList(list)
                    b.txtLogbookSummary.text =
                        if (list.size == 1) "1 ENTRY" else list.size.toString() + " ENTRIES"
                    b.emptyState.visibility = if (list.isEmpty()) View.VISIBLE else View.GONE
                    b.btnDeleteAll.isEnabled = list.isNotEmpty()
                }
        }

        runVerification()
    }

    // ---------------------------------------------------------------- chain verification

    private fun runVerification() {
        val bnd = _b ?: return
        bnd.txtChainDetail.text = "Verifying..."
        bnd.btnVerifyChain.isEnabled = false

        viewLifecycleOwner.lifecycleScope.launch {
            val status = ChainVerifier.verify(requireContext())
            val cur = _b ?: return@launch
            cur.btnVerifyChain.isEnabled = true

            when (status) {
                is ChainVerifier.Status.Empty -> {
                    cur.txtChainTitle.text = "SHA-256 chain"
                    cur.txtChainDetail.text = "No entries yet - nothing to verify"
                    tintChain(R.color.border, R.color.white, R.color.text_secondary)
                }

                is ChainVerifier.Status.Verified -> {
                    cur.txtChainTitle.text = "Chain verified"
                    val parts = ArrayList<String>()
                    parts.add(status.contentVerified.toString() + " of " +
                            status.total + " content-verified")
                    if (status.audioAbsent > 0) {
                        parts.add(status.audioAbsent.toString() + " link-only (audio removed)")
                    }
                    if (status.notChecked > 0) {
                        parts.add(status.notChecked.toString() + " not re-read (size limit)")
                    }
                    if (status.bridgedDeletions > 0) {
                        parts.add(status.bridgedDeletions.toString() + " audited deletion(s)")
                    }
                    cur.txtChainDetail.text = parts.joinToString("  ·  ")
                    tintChain(R.color.success, R.color.success_bg, R.color.text_secondary)
                }

                is ChainVerifier.Status.Broken -> {
                    cur.txtChainTitle.text = "Chain BROKEN at entry " +
                            status.position + " of " + status.total
                    cur.txtChainDetail.text = status.reason
                    tintChain(R.color.danger, R.color.danger_bg, R.color.danger)
                }
            }
        }
    }

    private fun tintChain(strokeRes: Int, bgRes: Int, detailRes: Int) {
        val cur = _b ?: return
        val ctx = context ?: return
        cur.cardChain.strokeColor = ContextCompat.getColor(ctx, strokeRes)
        cur.cardChain.setCardBackgroundColor(ContextCompat.getColor(ctx, bgRes))
        cur.txtChainDetail.setTextColor(ContextCompat.getColor(ctx, detailRes))
    }

    // ---------------------------------------------------------------- deletion

    private fun confirmDeleteOne(entry: LogbookEntry) {
        val extra = if (entry.syncState != "synced")
            "\n\nThis entry has NOT been uploaded yet. Deleting it loses the only copy."
        else ""
        MaterialAlertDialogBuilder(requireContext())
            .setTitle("Delete this recording?")
            .setMessage(
                Formats.timestamp(entry.startedAt) + "  ·  " +
                        Formats.duration(entry.durationSec) + "\n" +
                        "The audio is erased permanently. The deletion is written to the audit " +
                        "trail so the hash chain remains verifiable." + extra
            )
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Delete") { _, _ ->
                if (player?.isPlaying(entry.id) == true) player?.stop()
                viewLifecycleOwner.lifecycleScope.launch {
                    val r = LogbookRepository.deleteEntry(requireContext(), entry)
                    val cur = _b ?: return@launch
                    val msg = if (r.fileFailures > 0)
                        "Entry removed, but the audio file could not be deleted"
                    else "Recording deleted"
                    Snackbar.make(cur.root, msg, Snackbar.LENGTH_SHORT).show()
                    runVerification()
                }
            }
            .show()
    }

    private fun confirmDeleteAll() {
        viewLifecycleOwner.lifecycleScope.launch {
            val dao = AppDatabase.getDatabase(requireContext()).logbookEntryDao()
            val all = withContext(Dispatchers.IO) { dao.getAllEntries() }
            if (all.isEmpty()) return@launch
            val unsynced = all.count { it.syncState != "synced" }
            val bytes = withContext(Dispatchers.IO) {
                LogbookRepository.bytesOnDisk(requireContext())
            }

            val warn = if (unsynced > 0)
                "\n\n" + unsynced + " of these have NOT been uploaded. Deleting loses the only copy."
            else ""

            MaterialAlertDialogBuilder(requireContext())
                .setTitle("Delete all " + all.size + " recordings?")
                .setMessage(
                    "This frees " + Formats.bytes(bytes) + " and cannot be undone. Every " +
                            "deletion is written to the audit trail." + warn
                )
                .setNegativeButton("Cancel", null)
                .setPositiveButton("Delete all") { _, _ -> doDeleteAll() }
                .show()
        }
    }

    private fun doDeleteAll() {
        player?.stop()
        b.btnDeleteAll.isEnabled = false
        viewLifecycleOwner.lifecycleScope.launch {
            val r = LogbookRepository.deleteAll(requireContext())
            val cur = _b ?: return@launch
            val msg = if (r.fileFailures > 0)
                r.deleted.toString() + " removed, " + r.fileFailures + " file(s) could not be deleted"
            else r.deleted.toString() + " recordings deleted"
            Snackbar.make(cur.root, msg, Snackbar.LENGTH_LONG).show()
            runVerification()
        }
    }

    // ---------------------------------------------------------------- export

    /**
     * Decrypts to a WAV in the cache directory and hands it to the share sheet.
     *
     * The export is deliberately a copy in cache, not the stored file: the stored file is
     * ciphertext and is what the hash chain is computed over, so it must never leave through a
     * path that could modify it.
     */
    private fun exportAndShare(entry: LogbookEntry) {
        viewLifecycleOwner.lifecycleScope.launch {
            val wav = withContext(Dispatchers.IO) { player?.exportWav(entry) }
            val cur = _b ?: return@launch
            if (wav == null) {
                Snackbar.make(cur.root, "Audio missing or could not be decrypted", Snackbar.LENGTH_LONG).show()
                return@launch
            }
            val outFile = withContext(Dispatchers.IO) {
                val dir = File(requireContext().cacheDir, "exports").apply { mkdirs() }
                val f = File(dir, "logbook_" + entry.id.take(8) + ".wav")
                f.writeBytes(wav)
                f
            }
            val uri = FileProvider.getUriForFile(
                requireContext(),
                requireContext().packageName + ".fileprovider",
                outFile
            )
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "audio/wav"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(
                    Intent.EXTRA_SUBJECT,
                    "Logbook " + Formats.timestamp(entry.startedAt) + " (" + entry.classification + ")"
                )
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(send, "Export recording"))
        }
    }

    override fun onPause() {
        super.onPause()
        player?.stop()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        player?.release()
        player = null
        _b = null
    }
}
