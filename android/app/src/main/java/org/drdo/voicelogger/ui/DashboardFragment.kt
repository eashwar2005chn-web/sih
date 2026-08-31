package org.drdo.voicelogger.ui

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.HapticFeedbackConstants
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import org.drdo.voicelogger.R
import org.drdo.voicelogger.data.db.AppDatabase
import org.drdo.voicelogger.data.prefs.AppSettings
import org.drdo.voicelogger.data.repo.LogbookRepository
import org.drdo.voicelogger.databinding.FragmentDashboardBinding
import org.drdo.voicelogger.service.VoiceLoggerService
import org.drdo.voicelogger.storage.StorageLocationManager
import org.drdo.voicelogger.sync.SyncScheduler

/**
 * Live status screen.
 *
 * Everything shown here is polled from the running service or read from the database. There is
 * no placeholder text on this screen: when capture is stopped the meter reads empty and the
 * segment row reads "--", because that is the truth, rather than a plausible-looking number.
 */
class DashboardFragment : Fragment() {

    private var _b: FragmentDashboardBinding? = null
    private val b get() = _b!!

    private val ui = Handler(Looper.getMainLooper())
    private val sosHold = Handler(Looper.getMainLooper())
    private var sosArmed = false

    /** ~16 fps is enough for a level meter and costs almost nothing; the view interpolates. */
    private val tick = object : Runnable {
        override fun run() {
            refreshLive()
            ui.postDelayed(this, 60L)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, saved: Bundle?
    ): View {
        _b = FragmentDashboardBinding.inflate(inflater, container, false)
        return b.root
    }

    override fun onViewCreated(view: View, saved: Bundle?) {
        super.onViewCreated(view, saved)

        b.btnToggleService.setOnClickListener {
            val act = activity as? MainActivity ?: return@setOnClickListener
            if (VoiceLoggerService.isServiceRunning) {
                requireContext().startService(
                    Intent(requireContext(), VoiceLoggerService::class.java).apply {
                        action = VoiceLoggerService.ACTION_STOP
                    })
            } else {
                act.startLoggingService()
            }
            // The service flips its own flag asynchronously; re-read shortly after.
            ui.postDelayed({ refreshLive() }, 350L)
        }

        b.btnSyncNow.setOnClickListener { onSyncNow() }

        setupSosHold()

        viewLifecycleOwner.lifecycleScope.launch {
            val dao = AppDatabase.getDatabase(requireContext()).logbookEntryDao()
            dao.getEntryCountFlow().collectLatest { b.statTotal.text = it.toString() }
        }
        viewLifecycleOwner.lifecycleScope.launch {
            val dao = AppDatabase.getDatabase(requireContext()).logbookEntryDao()
            dao.getUnsyncedCountFlow().collectLatest { b.statQueued.text = it.toString() }
        }
        viewLifecycleOwner.lifecycleScope.launch {
            val dao = AppDatabase.getDatabase(requireContext()).logbookEntryDao()
            dao.getSyncedCountFlow().collectLatest { b.statSynced.text = it.toString() }
        }
    }

    /**
     * Hold-to-confirm, 1000 ms. A single tap must never dispatch an emergency alert, and a
     * hold that is released early must never dispatch one either - the callback is cancelled,
     * not merely checked afterwards.
     */
    private fun setupSosHold() {
        b.btnSosEmergency.setOnTouchListener { v, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    sosArmed = true
                    v.animate().scaleX(0.93f).scaleY(0.93f).setDuration(1000).start()
                    sosHold.postDelayed({
                        if (sosArmed) {
                            sosArmed = false
                            v.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
                            triggerSos()
                            v.animate().scaleX(1f).scaleY(1f).setDuration(180).start()
                        }
                    }, 1000L)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    val wasArmed = sosArmed
                    sosArmed = false
                    sosHold.removeCallbacksAndMessages(null)
                    v.animate().scaleX(1f).scaleY(1f).setDuration(180).start()
                    if (wasArmed && event.actionMasked == MotionEvent.ACTION_UP) {
                        Snackbar.make(b.root, "Hold for 1 second to dispatch SOS", Snackbar.LENGTH_SHORT).show()
                    }
                    true
                }
                else -> false
            }
        }
    }

    private fun triggerSos() {
        requireContext().startService(
            Intent(requireContext(), VoiceLoggerService::class.java).apply {
                action = VoiceLoggerService.ACTION_TRIGGER_SOS
            })
        (activity as? MainActivity)?.setSosVisible(true)
        Snackbar.make(b.root, "SOS dispatched. Recording continues.", Snackbar.LENGTH_LONG).show()
    }

    private fun onSyncNow() {
        val ctx = requireContext()
        if (!SyncScheduler.isConfigured(ctx)) {
            Snackbar.make(b.root, "Set a cloud URL in Settings first", Snackbar.LENGTH_LONG)
                .setAction("Settings") {
                    (activity as? MainActivity)?.findViewById<com.google.android.material.bottomnavigation.BottomNavigationView>(
                        R.id.bottomNav
                    )?.selectedItemId = R.id.nav_settings
                }.show()
            return
        }
        if (!AppSettings.isSyncEnabled(ctx)) {
            Snackbar.make(b.root, "Upload is switched off in Settings", Snackbar.LENGTH_LONG).show()
            return
        }
        SyncScheduler.syncNow(ctx)
        Snackbar.make(
            b.root,
            "Upload queued. It runs as soon as the network allows.",
            Snackbar.LENGTH_LONG
        ).show()
    }

    private fun refreshLive() {
        val bnd = _b ?: return
        val ctx = context ?: return
        val running = VoiceLoggerService.isServiceRunning

        bnd.dotLive.setBackgroundResource(
            if (running) R.drawable.bg_dot_live else R.drawable.bg_dot_idle
        )
        bnd.txtCaptureState.text = if (running) "Recording" else "Stopped"
        bnd.txtCaptureState.setTextColor(
            resources.getColor(if (running) R.color.success else R.color.text_muted, null)
        )
        bnd.btnToggleService.text = if (running) "Stop capture" else "Start capture"

        if (running) {
            val db = VoiceLoggerService.currentLevelDbfs
            bnd.levelMeter.setLevelDbfs(db)
            bnd.txtLevelDb.text = String.format(java.util.Locale.UK, "%.1f dB", db)

            val len = VoiceLoggerService.segmentLengthMs.coerceAtLeast(1L)
            val elapsed = (System.currentTimeMillis() - VoiceLoggerService.segmentStartedAtMs)
                .coerceIn(0L, len)
            bnd.segmentProgress.setProgressCompat(
                ((elapsed * 1000L) / len).toInt().coerceIn(0, 1000), true
            )
            bnd.txtSegmentElapsed.text =
                Formats.duration(elapsed / 1000.0) + " / " + Formats.duration(len / 1000.0)
            bnd.txtSegmentBuffered.text = Formats.bytes(VoiceLoggerService.bufferedBytes)
        } else {
            bnd.levelMeter.reset()
            bnd.txtLevelDb.text = "-60.0 dB"
            bnd.segmentProgress.setProgressCompat(0, false)
            bnd.txtSegmentElapsed.text = "--"
            bnd.txtSegmentBuffered.text = "--"
        }

        bnd.txtStorageDest.text = StorageLocationManager.describe(ctx)

        val ep = AppSettings.getCloudEndpoint(ctx)
        bnd.txtSyncState.text = when {
            ep.isBlank() -> "No endpoint configured"
            !AppSettings.isSyncEnabled(ctx) -> "Upload switched off"
            AppSettings.isUnmeteredOnly(ctx) -> ep + "  (Wi-Fi only)"
            else -> ep
        }

        (activity as? MainActivity)?.setSosVisible(VoiceLoggerService.sosActive)
    }

    override fun onResume() {
        super.onResume()
        ui.post(tick)
        // Walking the files is IO, so it runs once per resume rather than on the 60 ms tick.
        viewLifecycleOwner.lifecycleScope.launch {
            val used = LogbookRepository.bytesOnDisk(requireContext())
            _b?.txtStorageUsed?.text = Formats.bytes(used) + " on disk"
        }
    }

    override fun onPause() {
        super.onPause()
        ui.removeCallbacksAndMessages(null)
        sosHold.removeCallbacksAndMessages(null)
        sosArmed = false
    }

    override fun onDestroyView() {
        super.onDestroyView()
        ui.removeCallbacksAndMessages(null)
        sosHold.removeCallbacksAndMessages(null)
        _b = null
    }
}
