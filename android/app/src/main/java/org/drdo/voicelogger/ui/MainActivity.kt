package org.drdo.voicelogger.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import org.drdo.voicelogger.R
import org.drdo.voicelogger.databinding.ActivityMainBinding
import org.drdo.voicelogger.service.VoiceLoggerService
import org.drdo.voicelogger.sync.SyncScheduler

/**
 * Single-activity shell: header, SOS banner, three tabs.
 *
 * The SOS banner lives here rather than inside the dashboard so it stays on screen while the
 * operator is on any tab. An emergency that scrolls out of view is not an emergency indicator.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private val permissionRequest = 101

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.bottomNav.setOnItemSelectedListener { item ->
            val f: Fragment = when (item.itemId) {
                R.id.nav_logbook -> LogbookFragment()
                R.id.nav_settings -> SettingsFragment()
                else -> DashboardFragment()
            }
            val title = when (item.itemId) {
                R.id.nav_logbook -> getString(R.string.logbook_title)
                R.id.nav_settings -> getString(R.string.settings_title)
                else -> getString(R.string.app_subtitle)
            }
            binding.txtScreenTitle.text = title
            supportFragmentManager.beginTransaction()
                .setCustomAnimations(R.anim.fade_in, R.anim.fade_out)
                .replace(binding.navHost.id, f)
                .commit()
            true
        }

        if (savedInstanceState == null) {
            binding.bottomNav.selectedItemId = R.id.nav_dashboard
        }

        binding.btnStandDownSos.setOnClickListener {
            startService(Intent(this, VoiceLoggerService::class.java).apply {
                action = VoiceLoggerService.ACTION_STAND_DOWN_SOS
            })
            setSosVisible(false)
        }

        checkAndRequestPermissions()

        // Re-assert the standing upload schedule on every launch. KEEP policy means this is a
        // no-op when it is already scheduled, and it recovers the schedule if the app was
        // force-stopped (which cancels WorkManager jobs).
        SyncScheduler.ensurePeriodic(this)
    }

    override fun onResume() {
        super.onResume()
        setSosVisible(VoiceLoggerService.sosActive)
    }

    fun setSosVisible(visible: Boolean) {
        val v = binding.sosBanner
        if (visible && v.visibility != View.VISIBLE) {
            v.alpha = 0f
            v.visibility = View.VISIBLE
            v.animate().alpha(1f).setDuration(180).start()
        } else if (!visible && v.visibility == View.VISIBLE) {
            v.animate().alpha(0f).setDuration(140).withEndAction {
                v.visibility = View.GONE
            }.start()
        }
    }

    private fun checkAndRequestPermissions() {
        val wanted = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            wanted.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        val missing = wanted.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, missing.toTypedArray(), permissionRequest)
        } else {
            startLoggingService()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != permissionRequest) return
        // Only start capture if RECORD_AUDIO was actually granted. Starting a microphone
        // foreground service without it throws SecurityException on API 34+.
        val micIdx = permissions.indexOf(Manifest.permission.RECORD_AUDIO)
        val micOk = micIdx < 0 || grantResults.getOrNull(micIdx) == PackageManager.PERMISSION_GRANTED
        if (micOk) startLoggingService()
    }

    fun startLoggingService() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            checkAndRequestPermissions()
            return
        }
        val intent = Intent(this, VoiceLoggerService::class.java).apply {
            action = VoiceLoggerService.ACTION_START
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
    }
}
