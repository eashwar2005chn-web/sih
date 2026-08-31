package org.drdo.voicelogger.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.MotionEvent
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import org.drdo.voicelogger.data.db.AppDatabase
import org.drdo.voicelogger.databinding.ActivityMainBinding
import org.drdo.voicelogger.service.VoiceLoggerService

/**
 * Tactical Light-Minimalist UI for DRDO 24/7 Voice Logger.
 * Minimal, clean white theme (#ffffff ground, 8px grid, Inter typography).
 * Features Status Dashboard, SOS Emergency Button (800ms hold), Logbook timeline, and Health screen.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var db: AppDatabase

    private var sosPressStartTime = 0L
    private val PERMISSION_REQUEST_CODE = 101

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        db = AppDatabase.getDatabase(this)

        checkAndRequestPermissions()
        setupUIHandlers()
        observeData()
    }

    private fun checkAndRequestPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        val missing = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missing.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, missing.toTypedArray(), PERMISSION_REQUEST_CODE)
        } else {
            startLoggingService()
        }
    }

    private fun startLoggingService() {
        val intent = Intent(this, VoiceLoggerService::class.java).apply {
            action = VoiceLoggerService.ACTION_START
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
    }

    private fun setupUIHandlers() {
        // Service Toggle Button
        binding.btnToggleService.setOnClickListener {
            if (VoiceLoggerService.isServiceRunning) {
                val intent = Intent(this, VoiceLoggerService::class.java).apply {
                    action = VoiceLoggerService.ACTION_STOP
                }
                startService(intent)
                binding.btnToggleService.text = "Start 24/7 Service"
            } else {
                startLoggingService()
                binding.btnToggleService.text = "Pause Service"
            }
        }

        // SOS Emergency Button — Press and Hold ~800ms
        binding.btnSosEmergency.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    sosPressStartTime = System.currentTimeMillis()
                    binding.btnSosEmergency.animate().scaleX(0.95f).scaleY(0.95f).setDuration(150).start()
                    true
                }
                MotionEvent.ACTION_UP -> {
                    val duration = System.currentTimeMillis() - sosPressStartTime
                    binding.btnSosEmergency.animate().scaleX(1.0f).scaleY(1.0f).setDuration(150).start()
                    if (duration >= 800) {
                        triggerSosAlert()
                    } else {
                        Toast.makeText(this, "Press and hold SOS button for 1 second to confirm alert.", Toast.LENGTH_SHORT).show()
                    }
                    true
                }
                MotionEvent.ACTION_CANCEL -> {
                    binding.btnSosEmergency.animate().scaleX(1.0f).scaleY(1.0f).setDuration(150).start()
                    true
                }
                else -> false
            }
        }

        binding.btnStandDownSos.setOnClickListener {
            val intent = Intent(this, VoiceLoggerService::class.java).apply {
                action = VoiceLoggerService.ACTION_STAND_DOWN_SOS
            }
            startService(intent)
            binding.sosBanner.visibility = View.GONE
        }
    }

    private fun triggerSosAlert() {
        val intent = Intent(this, VoiceLoggerService::class.java).apply {
            action = VoiceLoggerService.ACTION_TRIGGER_SOS
        }
        startService(intent)

        binding.sosBanner.visibility = View.VISIBLE
        Toast.makeText(this, "🚨 EMERGENCY SOS DISPATCHED — Recording Unbroken", Toast.LENGTH_LONG).show()
    }

    private fun observeData() {
        lifecycleScope.launch {
            db.logbookEntryDao().getUnsyncedCountFlow().collectLatest { count ->
                binding.txtQueueCount.text = "$count entries queued for sync"
            }
        }

        lifecycleScope.launch {
            db.logbookEntryDao().getAllEntriesFlow().collectLatest { entries ->
                binding.txtLogbookCount.text = "${entries.size} Total Logbook Entries"
            }
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            startLoggingService()
        }
    }
}
