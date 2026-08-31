package org.drdo.voicelogger.ui

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.drdo.voicelogger.BuildConfig
import org.drdo.voicelogger.R
import org.drdo.voicelogger.data.db.AppDatabase
import org.drdo.voicelogger.data.prefs.AppSettings
import org.drdo.voicelogger.data.repo.LogbookRepository
import org.drdo.voicelogger.databinding.FragmentSettingsBinding
import org.drdo.voicelogger.storage.StorageLocationManager
import org.drdo.voicelogger.sync.SyncScheduler
import java.net.HttpURLConnection
import java.net.URL

/**
 * Configuration: where recordings are written, where they are uploaded, how long a segment is,
 * and bulk deletion.
 */
class SettingsFragment : Fragment() {

    private var _b: FragmentSettingsBinding? = null
    private val b get() = _b!!

    /**
     * SAF directory picker. A persistable grant is taken immediately, which is what lets the
     * background service keep writing there after a reboot; without takePersistableUriPermission
     * the grant dies with this Activity and capture would silently revert to internal storage.
     */
    private val pickFolder = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri: Uri? ->
        if (uri == null) return@registerForActivityResult
        // The picker result can land after this view is gone (rotation, process restart while
        // the chooser was open), so the setting is persisted from the application context and
        // every view touch goes through _b, never the non-null b.
        val app = requireContext().applicationContext
        val ok = runCatching {
            app.contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
            )
        }.isSuccess

        if (!ok) {
            _b?.let {
                Snackbar.make(it.root, "Could not keep access to that folder", Snackbar.LENGTH_LONG).show()
            }
            return@registerForActivityResult
        }
        AppSettings.setStorageTreeUri(app, uri.toString())
        refreshStorage()
        _b?.let {
            Snackbar.make(
                it.root,
                "New recordings will be saved to " + StorageLocationManager.describe(app),
                Snackbar.LENGTH_LONG
            ).show()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, saved: Bundle?
    ): View {
        _b = FragmentSettingsBinding.inflate(inflater, container, false)
        return b.root
    }

    override fun onViewCreated(view: View, saved: Bundle?) {
        super.onViewCreated(view, saved)
        val ctx = requireContext()

        // ---- storage ----
        b.btnPickFolder.setOnClickListener {
            runCatching { pickFolder.launch(null) }.onFailure {
                Snackbar.make(b.root, "No folder picker available on this device", Snackbar.LENGTH_LONG).show()
            }
        }
        b.btnUseInternal.setOnClickListener {
            AppSettings.setStorageTreeUri(ctx, null)
            refreshStorage()
            Snackbar.make(b.root, "New recordings will use internal storage", Snackbar.LENGTH_SHORT).show()
        }

        // ---- cloud ----
        b.editEndpoint.setText(AppSettings.getCloudEndpoint(ctx))
        b.swSyncEnabled.isChecked = AppSettings.isSyncEnabled(ctx)
        b.swUnmetered.isChecked = AppSettings.isUnmeteredOnly(ctx)
        b.swDeleteAfterSync.isChecked = AppSettings.isDeleteAfterSync(ctx)

        b.btnSaveEndpoint.setOnClickListener { saveEndpoint() }
        b.btnTestEndpoint.setOnClickListener { testEndpoint() }

        b.swSyncEnabled.setOnCheckedChangeListener { _, v ->
            AppSettings.setSyncEnabled(ctx, v)
            SyncScheduler.reschedule(ctx)
        }
        b.swUnmetered.setOnCheckedChangeListener { _, v ->
            AppSettings.setUnmeteredOnly(ctx, v)
            SyncScheduler.reschedule(ctx)
        }
        b.swDeleteAfterSync.setOnCheckedChangeListener { _, v ->
            if (v) confirmDeleteAfterSync() else AppSettings.setDeleteAfterSync(ctx, false)
        }

        // ---- capture ----
        b.editSegmentMinutes.setText(AppSettings.getSegmentMinutes(ctx).toString())
        b.editDeviceId.setText(AppSettings.getDeviceId(ctx))
        b.btnSaveCapture.setOnClickListener { saveCapture() }

        // ---- danger ----
        b.btnPurgeAll.setOnClickListener { confirmPurge() }

        b.txtBuildInfo.text = "v" + BuildConfig.VERSION_NAME + "  ·  PS 26052"

        refreshStorage()
    }

    private fun refreshStorage() {
        val cur = _b ?: return
        val ctx = context ?: return
        cur.txtStorageCurrent.text = StorageLocationManager.describe(ctx)
        viewLifecycleOwner.lifecycleScope.launch {
            val used = LogbookRepository.bytesOnDisk(ctx)
            val count = withContext(Dispatchers.IO) {
                AppDatabase.getDatabase(ctx).logbookEntryDao().getAllEntries().size
            }
            _b?.txtStorageUsage?.text =
                Formats.bytes(used) + " across " + count + " recording(s)"
        }
    }

    // ------------------------------------------------------------------ cloud

    private fun saveEndpoint() {
        val ctx = requireContext()
        val raw = b.editEndpoint.text?.toString()?.trim().orEmpty()

        if (raw.isNotEmpty() && !(raw.startsWith("http://") || raw.startsWith("https://"))) {
            b.tilEndpoint.error = "Must start with http:// or https://"
            return
        }
        b.tilEndpoint.error = null
        AppSettings.setCloudEndpoint(ctx, raw)
        SyncScheduler.reschedule(ctx)

        val msg = if (raw.isEmpty()) "Endpoint cleared - uploads disabled"
        else "Endpoint saved"
        Snackbar.make(b.root, msg, Snackbar.LENGTH_SHORT).show()

        if (raw.isNotEmpty() && raw.startsWith("http://")) {
            Snackbar.make(
                b.root,
                "Plain HTTP is unencrypted in transit. Prefer https for field use.",
                Snackbar.LENGTH_LONG
            ).show()
        }
    }

    /**
     * Issues a real request and reports exactly what came back - status line or exception text.
     * Never reports "OK" for anything that was not actually reached, and a non-2xx response is
     * shown as-is rather than being smoothed into a pass.
     */
    private fun testEndpoint() {
        val raw = b.editEndpoint.text?.toString()?.trim().orEmpty().trimEnd('/')
        if (raw.isEmpty()) {
            showTestResult("No URL entered", R.color.text_muted)
            return
        }
        if (!(raw.startsWith("http://") || raw.startsWith("https://"))) {
            showTestResult("URL must start with http:// or https://", R.color.danger)
            return
        }

        b.btnTestEndpoint.isEnabled = false
        showTestResult("Contacting " + raw + " ...", R.color.text_secondary)

        viewLifecycleOwner.lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                var conn: HttpURLConnection? = null
                try {
                    conn = URL(raw).openConnection() as HttpURLConnection
                    conn.requestMethod = "GET"
                    conn.connectTimeout = 8000
                    conn.readTimeout = 8000
                    val code = conn.responseCode
                    val msg = conn.responseMessage ?: ""
                    Pair(code in 200..399, "HTTP " + code + " " + msg)
                } catch (e: Exception) {
                    Pair(false, e.javaClass.simpleName + ": " + (e.message ?: "no detail"))
                } finally {
                    conn?.disconnect()
                }
            }
            _b?.btnTestEndpoint?.isEnabled = true
            showTestResult(
                result.second,
                if (result.first) R.color.success else R.color.danger
            )
        }
    }

    private fun showTestResult(text: String, colourRes: Int) {
        val cur = _b ?: return
        val ctx = context ?: return
        cur.txtEndpointResult.visibility = View.VISIBLE
        cur.txtEndpointResult.text = text
        cur.txtEndpointResult.setTextColor(ContextCompat.getColor(ctx, colourRes))
    }

    private fun confirmDeleteAfterSync() {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle("Delete local audio after upload?")
            .setMessage(
                "Once a NORMAL recording is uploaded its audio is erased from this device. " +
                        "The logbook row and its hashes are kept, so the chain still verifies by " +
                        "link - but the recording can no longer be played back or content-verified " +
                        "here.\n\nSOS and protected recordings are never auto-deleted."
            )
            .setNegativeButton("Cancel") { _, _ ->
                _b?.swDeleteAfterSync?.isChecked = false
            }
            .setOnCancelListener { _b?.swDeleteAfterSync?.isChecked = false }
            .setPositiveButton("Enable") { _, _ ->
                AppSettings.setDeleteAfterSync(requireContext(), true)
            }
            .show()
    }

    // ------------------------------------------------------------------ capture

    private fun saveCapture() {
        val ctx = requireContext()
        val mins = b.editSegmentMinutes.text?.toString()?.trim()?.toIntOrNull()
        if (mins == null || mins < 1 || mins > 720) {
            Snackbar.make(b.root, "Segment length must be 1 to 720 minutes", Snackbar.LENGTH_LONG).show()
            return
        }
        AppSettings.setSegmentMinutes(ctx, mins)
        AppSettings.setDeviceId(ctx, b.editDeviceId.text?.toString().orEmpty())
        b.editDeviceId.setText(AppSettings.getDeviceId(ctx))

        Snackbar.make(
            b.root,
            "Saved. The new segment length applies from the next capture start.",
            Snackbar.LENGTH_LONG
        ).show()
    }

    // ------------------------------------------------------------------ danger

    private fun confirmPurge() {
        viewLifecycleOwner.lifecycleScope.launch {
            val ctx = requireContext()
            val dao = AppDatabase.getDatabase(ctx).logbookEntryDao()
            val all = withContext(Dispatchers.IO) { dao.getAllEntries() }
            if (all.isEmpty()) {
                Snackbar.make(b.root, "There are no recordings to delete", Snackbar.LENGTH_SHORT).show()
                return@launch
            }
            val unsynced = all.count { it.syncState != "synced" }
            val bytes = withContext(Dispatchers.IO) { LogbookRepository.bytesOnDisk(ctx) }
            val warn = if (unsynced > 0)
                "\n\n" + unsynced + " have NOT been uploaded. Deleting loses the only copy."
            else ""

            MaterialAlertDialogBuilder(ctx)
                .setTitle("Delete all " + all.size + " recordings?")
                .setMessage(
                    "This frees " + Formats.bytes(bytes) + " and cannot be undone." + warn
                )
                .setNegativeButton("Cancel", null)
                .setPositiveButton("Delete all") { _, _ -> doPurge() }
                .show()
        }
    }

    private fun doPurge() {
        val ctx = requireContext()
        viewLifecycleOwner.lifecycleScope.launch {
            val r = LogbookRepository.deleteAll(ctx)
            val cur = _b ?: return@launch
            val tail = if (r.fileFailures > 0)
                ", " + r.fileFailures + " file(s) could not be deleted" else ""
            Snackbar.make(
                cur.root,
                r.deleted.toString() + " recordings deleted" + tail,
                Snackbar.LENGTH_LONG
            ).show()
            refreshStorage()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _b = null
    }
}
