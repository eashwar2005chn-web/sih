package org.drdo.voicelogger.sync

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.lifecycle.LiveData
import org.drdo.voicelogger.data.prefs.AppSettings
import java.util.concurrent.TimeUnit

/**
 * Schedules [ChunkedSyncWorker].
 *
 * The worker was fully written but NEVER ENQUEUED anywhere in the app - no WorkManager call
 * existed in the codebase. The dashboard's "N entries queued for sync" was therefore literally
 * true and permanently unactionable: entries queued and stayed queued forever. This closes
 * PS requirement 3 (local capture -> cloud on reconnect).
 *
 * Two entry points, deliberately distinct:
 *
 *   [ensurePeriodic] - the standing job. KEEP policy, so it is safe to call on every service
 *                      start and every app launch without resetting its schedule.
 *   [syncNow]        - a one-shot, used after a segment rotates and by the manual button.
 *                      REPLACE policy so a fresh request supersedes a stale queued one.
 *
 * Both carry a CONNECTED (or UNMETERED) constraint, which is what makes the offline-first
 * story work: with no network the request simply waits, and WorkManager runs it the moment
 * connectivity returns, across process death and reboot.
 */
object SyncScheduler {

    const val PERIODIC_WORK = "drdo_sync_periodic"
    const val ONESHOT_WORK = "drdo_sync_oneshot"

    private fun constraints(ctx: Context) = Constraints.Builder()
        .setRequiredNetworkType(
            if (AppSettings.isUnmeteredOnly(ctx)) NetworkType.UNMETERED else NetworkType.CONNECTED
        )
        .build()

    /**
     * True when sync can actually do something. An unset endpoint is a configuration state,
     * not an error, and must never be reported to the operator as a sync failure.
     */
    fun isConfigured(ctx: Context): Boolean {
        val ep = AppSettings.getCloudEndpoint(ctx)
        return ep.isNotBlank() && (ep.startsWith("http://") || ep.startsWith("https://"))
    }

    fun ensurePeriodic(ctx: Context) {
        val wm = WorkManager.getInstance(ctx.applicationContext)
        if (!AppSettings.isSyncEnabled(ctx) || !isConfigured(ctx)) {
            wm.cancelUniqueWork(PERIODIC_WORK)
            return
        }
        // 15 min is WorkManager's floor for periodic work; anything smaller is silently raised.
        val req = PeriodicWorkRequestBuilder<ChunkedSyncWorker>(15, TimeUnit.MINUTES)
            .setConstraints(constraints(ctx))
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .build()
        wm.enqueueUniquePeriodicWork(PERIODIC_WORK, ExistingPeriodicWorkPolicy.KEEP, req)
    }

    /** Fire a sync attempt as soon as the network constraint allows. */
    fun syncNow(ctx: Context) {
        if (!AppSettings.isSyncEnabled(ctx) || !isConfigured(ctx)) return
        val req = OneTimeWorkRequestBuilder<ChunkedSyncWorker>()
            .setConstraints(constraints(ctx))
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 15, TimeUnit.SECONDS)
            .build()
        WorkManager.getInstance(ctx.applicationContext)
            .enqueueUniqueWork(ONESHOT_WORK, ExistingWorkPolicy.REPLACE, req)
    }

    /** Re-apply both schedules after a settings change (endpoint, metering, enable/disable). */
    fun reschedule(ctx: Context) {
        val wm = WorkManager.getInstance(ctx.applicationContext)
        wm.cancelUniqueWork(PERIODIC_WORK)
        ensurePeriodic(ctx)
    }

    fun cancelAll(ctx: Context) {
        val wm = WorkManager.getInstance(ctx.applicationContext)
        wm.cancelUniqueWork(PERIODIC_WORK)
        wm.cancelUniqueWork(ONESHOT_WORK)
    }

    fun observeOneShot(ctx: Context): LiveData<List<WorkInfo>> =
        WorkManager.getInstance(ctx.applicationContext)
            .getWorkInfosForUniqueWorkLiveData(ONESHOT_WORK)
}
