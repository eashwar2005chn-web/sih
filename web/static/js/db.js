/* ============================================================================
   DRDO PS 26052 — IndexedDB Database Layer
   Durable client storage for megabyte audio blobs & metadata.
   ========================================================================== */

const DB_NAME = 'anc_field_db';
const DB_VERSION = 1;
const STORE_RECORDINGS = 'recordings';
const STORE_META = 'meta';

class ANCFieldDB {
  constructor() {
    this.db = null;
  }

  async init() {
    if (this.db) return this.db;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        // 1. Recordings Store
        if (!db.objectStoreNames.contains(STORE_RECORDINGS)) {
          const recStore = db.createObjectStore(STORE_RECORDINGS, { keyPath: 'id' });
          recStore.createIndex('syncState', 'syncState', { unique: false });
          recStore.createIndex('createdAt', 'createdAt', { unique: false });
        }

        // 2. Meta/Settings Store
        if (!db.objectStoreNames.contains(STORE_META)) {
          db.createObjectStore(STORE_META, { keyPath: 'key' });
        }
      };

      request.onsuccess = (event) => {
        this.db = event.target.result;
        resolve(this.db);
      };

      request.onerror = (event) => {
        console.error('[DB] IndexedDB opening error:', event.target.error);
        reject(event.target.error);
      };
    });
  }

  // --- Storage Quota Check ---
  async checkStorageQuota() {
    if (navigator.storage && navigator.storage.estimate) {
      try {
        const estimate = await navigator.storage.estimate();
        const usage = estimate.usage || 0;
        const quota = estimate.quota || 1;
        const percent = Math.round((usage / quota) * 100);
        const freeMB = Math.round((quota - usage) / (1024 * 1024));

        return {
          usageMB: Math.round(usage / (1024 * 1024)),
          quotaMB: Math.round(quota / (1024 * 1024)),
          freeMB: freeMB,
          percentUsed: percent,
          isCritical: percent > 90 || freeMB < 10
        };
      } catch (err) {
        console.warn('[DB] Could not estimate storage:', err);
      }
    }
    return { usageMB: 0, quotaMB: 0, freeMB: 500, percentUsed: 0, isCritical: false };
  }

  // --- Recording CRUD ---
  async saveRecording(recording) {
    await this.init();

    // Check storage quota before writing large blob
    const quota = await this.checkStorageQuota();
    if (quota.isCritical) {
      throw new Error(`Device storage critically low (${quota.freeMB} MB free, ${quota.percentUsed}% used). Clear local data or sync first.`);
    }

    return new Promise((resolve, reject) => {
      const tx = this.db.transaction([STORE_RECORDINGS], 'readwrite');
      const store = tx.objectStore(STORE_RECORDINGS);

      const record = {
        id: recording.id || self.crypto.randomUUID(),
        blob: recording.blob,
        mimeType: recording.mimeType || 'audio/wav',
        durationSec: recording.durationSec || 0,
        sampleRate: recording.sampleRate || 16000,
        createdAt: recording.createdAt || new Date().toISOString(),
        syncState: recording.syncState || 'local', // 'local' | 'queued' | 'uploading' | 'synced' | 'failed'
        serverId: recording.serverId || null,
        attempts: recording.attempts || 0,
        lastError: recording.lastError || null
      };

      const req = store.put(record);
      req.onsuccess = () => resolve(record);
      req.onerror = (e) => reject(e.target.error);
    });
  }

  async getRecording(id) {
    await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction([STORE_RECORDINGS], 'readonly');
      const store = tx.objectStore(STORE_RECORDINGS);
      const req = store.get(id);
      req.onsuccess = () => resolve(req.result);
      req.onerror = (e) => reject(e.target.error);
    });
  }

  async getAllRecordings() {
    await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction([STORE_RECORDINGS], 'readonly');
      const store = tx.objectStore(STORE_RECORDINGS);
      const index = store.index('createdAt');
      const req = index.getAll();
      req.onsuccess = () => {
        // Sort descending by createdAt
        const items = req.result || [];
        items.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
        resolve(items);
      };
      req.onerror = (e) => reject(e.target.error);
    });
  }

  async getUnsyncedRecordings() {
    await this.init();
    const all = await this.getAllRecordings();
    return all.filter(item => item.syncState !== 'synced');
  }

  async updateRecordingState(id, updates) {
    await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction([STORE_RECORDINGS], 'readwrite');
      const store = tx.objectStore(STORE_RECORDINGS);
      const getReq = store.get(id);

      getReq.onsuccess = () => {
        const existing = getReq.result;
        if (!existing) {
          return reject(new Error(`Recording '${id}' not found in IndexedDB.`));
        }

        const updated = { ...existing, ...updates };
        const putReq = store.put(updated);
        putReq.onsuccess = () => resolve(updated);
        putReq.onerror = (e) => reject(e.target.error);
      };
      getReq.onerror = (e) => reject(e.target.error);
    });
  }

  async deleteRecording(id) {
    await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction([STORE_RECORDINGS], 'readwrite');
      const store = tx.objectStore(STORE_RECORDINGS);
      const req = store.delete(id);
      req.onsuccess = () => resolve(true);
      req.onerror = (e) => reject(e.target.error);
    });
  }

  async clearAllRecordings() {
    await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction([STORE_RECORDINGS], 'readwrite');
      const store = tx.objectStore(STORE_RECORDINGS);
      const req = store.clear();
      req.onsuccess = () => resolve(true);
      req.onerror = (e) => reject(e.target.error);
    });
  }

  // --- Settings Meta CRUD ---
  async getMeta(key, defaultValue = null) {
    await this.init();
    return new Promise((resolve) => {
      const tx = this.db.transaction([STORE_META], 'readonly');
      const store = tx.objectStore(STORE_META);
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result ? req.result.value : defaultValue);
      req.onerror = () => resolve(defaultValue);
    });
  }

  async setMeta(key, value) {
    await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction([STORE_META], 'readwrite');
      const store = tx.objectStore(STORE_META);
      const req = store.put({ key, value });
      req.onsuccess = () => resolve(value);
      req.onerror = (e) => reject(e.target.error);
    });
  }
}

// Global Singleton Export
window.ancDB = new ANCFieldDB();
