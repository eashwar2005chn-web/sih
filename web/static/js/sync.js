/* ============================================================================
   DRDO PS 26052 — Sync Engine
   Serial, idempotent background sync with exponential backoff & resume.
   ========================================================================== */

class SyncEngine {
  constructor() {
    this.isSyncing = false;
    this.syncListeners = [];
    this.maxBackoffMs = 60000; // Cap at 60s
  }

  init() {
    // 1. Listen for browser online event
    window.addEventListener('online', () => {
      console.log('[Sync] Network online detected. Triggering sync...');
      this.triggerSync('network_online');
    });

    window.addEventListener('offline', () => {
      console.log('[Sync] Network offline detected.');
      this.notifyListeners();
    });

    // 2. Service Worker Message Listener
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.addEventListener('message', (event) => {
        if (event.data && event.data.type === 'TRIGGER_BACKGROUND_SYNC') {
          console.log('[Sync] Service worker triggered background sync');
          this.triggerSync('background_sync');
        }
      });
    }

    // 3. Register Background Sync if supported
    this.registerBackgroundSync();

    // 4. Initial check
    if (navigator.onLine) {
      setTimeout(() => this.triggerSync('init'), 1500);
    }
  }

  async registerBackgroundSync() {
    if ('serviceWorker' in navigator && 'SyncManager' in window) {
      try {
        const reg = await navigator.serviceWorker.ready;
        await reg.sync.register('sync-recordings');
        console.log('[Sync] Background Sync registered successfully');
      } catch (err) {
        console.warn('[Sync] Background Sync registration failed (will use online event fallback):', err);
      }
    }
  }

  subscribe(callback) {
    this.syncListeners.push(callback);
  }

  notifyListeners() {
    this.syncListeners.forEach(cb => cb());
  }

  async getEndpointUrl() {
    const metaEndpoint = await window.ancDB.getMeta('endpointUrl', '/api/recordings');
    return metaEndpoint || '/api/recordings';
  }

  // Blob to Base64 Helper
  blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  async triggerSync(source = 'manual') {
    if (this.isSyncing) {
      console.log('[Sync] Sync already in progress. Skipping trigger from:', source);
      return;
    }

    if (!navigator.onLine) {
      console.log('[Sync] Device is offline. Cannot sync right now.');
      this.notifyListeners();
      return;
    }

    this.isSyncing = true;
    this.notifyListeners();

    try {
      const endpoint = await this.getEndpointUrl();
      const unsynced = await window.ancDB.getUnsyncedRecordings();

      if (unsynced.length === 0) {
        console.log('[Sync] Queue is empty. Everything is up to date.');
        await window.ancDB.setMeta('lastSyncTime', new Date().toISOString());
        this.isSyncing = false;
        this.notifyListeners();
        return;
      }

      console.log(`[Sync] Starting upload queue (${unsynced.length} items waiting)...`);

      // Process oldest first (unsynced is already sorted by createdAt desc, reverse to get oldest first)
      const queue = [...unsynced].reverse();

      for (const item of queue) {
        if (!navigator.onLine) {
          console.warn('[Sync] Lost connectivity mid-queue. Pausing sync engine.');
          break;
        }

        // Mark as uploading
        await window.ancDB.updateRecordingState(item.id, { syncState: 'uploading' });
        this.notifyListeners();

        try {
          // Convert audio Blob to Base64
          const audioB64 = await this.blobToBase64(item.blob);

          const payload = {
            id: item.id,
            mimeType: item.mimeType,
            durationSec: item.durationSec,
            sampleRate: item.sampleRate,
            createdAt: item.createdAt,
            audio_b64: audioB64
          };

          const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
          });

          if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Server returned HTTP ${response.status}: ${errText}`);
          }

          const result = await response.json();

          // Strictly verify server confirmation before marking synced!
          if (result.status === 'created' || result.status === 'already_exists') {
            console.log(`[Sync] Item ${item.id} confirmed by server (${result.status}).`);
            await window.ancDB.updateRecordingState(item.id, {
              syncState: 'synced',
              serverId: result.id,
              lastError: null
            });
          } else {
            throw new Error(result.error || 'Server did not return confirmation status.');
          }

        } catch (uploadError) {
          console.error(`[Sync] Failed uploading recording ${item.id}:`, uploadError);
          const attempts = (item.attempts || 0) + 1;
          const backoffDelay = Math.min(this.maxBackoffMs, 1000 * Math.pow(2, attempts));

          await window.ancDB.updateRecordingState(item.id, {
            syncState: 'failed',
            attempts: attempts,
            lastError: `${uploadError.message} (Next retry in ${Math.round(backoffDelay / 1000)}s)`
          });

          // Wait backoff time before processing next item in serial queue
          await new Promise(res => setTimeout(res, 1000));
        }
      }

      await window.ancDB.setMeta('lastSyncTime', new Date().toISOString());

    } catch (globalError) {
      console.error('[Sync] Fatal sync engine error:', globalError);
    } finally {
      this.isSyncing = false;
      this.notifyListeners();
    }
  }
}

// Global Singleton Export
window.syncEngine = new SyncEngine();
