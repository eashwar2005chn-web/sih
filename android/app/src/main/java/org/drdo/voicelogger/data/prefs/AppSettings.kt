package org.drdo.voicelogger.data.prefs

import android.content.Context
import android.content.SharedPreferences

/**
 * Typed wrapper over SharedPreferences for every user-configurable setting.
 *
 * Deliberately a plain object rather than DataStore: the service reads these values from a
 * non-suspending audio loop, and a blocking read of an already-loaded SharedPreferences map is
 * the correct tool for that. DataStore would force a runBlocking on the capture thread.
 */
object AppSettings {

    private const val PREFS = "drdo_voicelogger_settings"

    private const val K_STORAGE_TREE_URI = "storage_tree_uri"
    private const val K_CLOUD_ENDPOINT = "cloud_endpoint"
    private const val K_SYNC_ENABLED = "sync_enabled"
    private const val K_UNMETERED_ONLY = "unmetered_only"
    private const val K_SEGMENT_MINUTES = "segment_minutes"
    private const val K_DEVICE_ID = "device_id"
    private const val K_DELETE_AFTER_SYNC = "delete_after_sync"
    private const val K_KEEP_SCREEN_STATS = "keep_screen_stats"

    /** No default endpoint. An unset endpoint means "not configured", never a silent fallback. */
    const val DEFAULT_ENDPOINT = ""
    const val DEFAULT_SEGMENT_MINUTES = 60
    const val DEFAULT_DEVICE_ID = "ANDROID_FIELD_UNIT_01"

    private fun p(ctx: Context): SharedPreferences =
        ctx.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /** Persisted SAF tree URI, or null when segments go to app-private internal storage. */
    fun getStorageTreeUri(ctx: Context): String? =
        p(ctx).getString(K_STORAGE_TREE_URI, null)

    fun setStorageTreeUri(ctx: Context, uri: String?) =
        p(ctx).edit().apply { if (uri == null) remove(K_STORAGE_TREE_URI) else putString(K_STORAGE_TREE_URI, uri) }.apply()

    fun getCloudEndpoint(ctx: Context): String =
        p(ctx).getString(K_CLOUD_ENDPOINT, DEFAULT_ENDPOINT) ?: DEFAULT_ENDPOINT

    fun setCloudEndpoint(ctx: Context, v: String) =
        p(ctx).edit().putString(K_CLOUD_ENDPOINT, v.trim()).apply()

    fun isSyncEnabled(ctx: Context): Boolean = p(ctx).getBoolean(K_SYNC_ENABLED, true)
    fun setSyncEnabled(ctx: Context, v: Boolean) = p(ctx).edit().putBoolean(K_SYNC_ENABLED, v).apply()

    fun isUnmeteredOnly(ctx: Context): Boolean = p(ctx).getBoolean(K_UNMETERED_ONLY, false)
    fun setUnmeteredOnly(ctx: Context, v: Boolean) = p(ctx).edit().putBoolean(K_UNMETERED_ONLY, v).apply()

    fun getSegmentMinutes(ctx: Context): Int =
        p(ctx).getInt(K_SEGMENT_MINUTES, DEFAULT_SEGMENT_MINUTES).coerceAtLeast(1)

    fun setSegmentMinutes(ctx: Context, v: Int) =
        p(ctx).edit().putInt(K_SEGMENT_MINUTES, v.coerceIn(1, 720)).apply()

    fun getDeviceId(ctx: Context): String =
        p(ctx).getString(K_DEVICE_ID, DEFAULT_DEVICE_ID) ?: DEFAULT_DEVICE_ID

    fun setDeviceId(ctx: Context, v: String) =
        p(ctx).edit().putString(K_DEVICE_ID, v.trim().ifEmpty { DEFAULT_DEVICE_ID }).apply()

    /**
     * Off by default, and deliberately so. Deleting local audio the moment a server acknowledges
     * it makes the upload unverifiable after the fact; the operator opts in.
     */
    fun isDeleteAfterSync(ctx: Context): Boolean = p(ctx).getBoolean(K_DELETE_AFTER_SYNC, false)
    fun setDeleteAfterSync(ctx: Context, v: Boolean) = p(ctx).edit().putBoolean(K_DELETE_AFTER_SYNC, v).apply()

    fun isKeepScreenOn(ctx: Context): Boolean = p(ctx).getBoolean(K_KEEP_SCREEN_STATS, false)
    fun setKeepScreenOn(ctx: Context, v: Boolean) = p(ctx).edit().putBoolean(K_KEEP_SCREEN_STATS, v).apply()
}
