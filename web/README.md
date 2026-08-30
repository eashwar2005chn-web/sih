# DRDO PS 26052 — Offline-First PWA Field Recording Application

This directory contains the **offline-first Progressive Web App (PWA)** for DRDO Problem Statement 26052, designed for field operators in low-connectivity or zero-connectivity tactical environments.

---

## Key Features

1. **100% Offline Field Recording**: Record audio directly from the device microphone in airplane mode. Zero network access required during recording.
2. **Durable Local Storage (IndexedDB)**: Audio recordings and metadata are stored in IndexedDB (`anc_field_db`), avoiding the 5 MB string limit of `localStorage`.
3. **PWA App Shell (Service Worker)**: Caches HTML, CSS, JS, and font assets (`anc-field-v1`) with a cache-first strategy. Offline loading is fully supported.
4. **Background Sync Engine**: Automatically uploads queued recordings serially (oldest first) to the backend server (`POST /api/recordings`) with exponential backoff on network failures.
5. **Idempotent SQLite Backend**: Client UUIDs prevent duplicate rows when re-syncing.
6. **Strict Honesty & State Tracking**: Records only transition to `synced` after receiving explicit HTTP confirmation from the backend.

---

## How to Run the Backend Server

The backend runs on Python standard library `http.server` with zero external web framework dependencies (no Flask/FastAPI required).

```bash
# Run server on port 8080 (default)
python web/server.py

# Or specify a custom port
python web/server.py 8080
```

Access the application in your web browser at:
`http://localhost:8080`

---

## How to Test Offline Behaviour & Sync Recovery

Follow these steps to verify offline recording and background sync recovery using Chrome DevTools:

### Step 1: Initial Load & PWA Caching
1. Open `http://localhost:8080` in Google Chrome or Microsoft Edge.
2. Open **DevTools** (`F12` or `Ctrl+Shift+I`) and navigate to the **Application** tab.
3. Under **Service Workers**, confirm `sw.js` is registered and active.
4. Optionally click **"Install App"** in the top header to install the PWA standalone shell.

### Step 2: Simulate Offline Mode
1. In DevTools, switch to the **Network** tab.
2. Click the throttling dropdown (currently set to "No throttling") and select **"Offline"** (or turn on Airplane Mode).
3. Navigate to the **🎙 Field Recorder** tab.
4. Click the giant red microphone button to start recording. (Grant microphone permission when prompted).
5. Speak into the microphone for 5 seconds.
6. Click the microphone button again to stop recording.
7. Observe the status message: `"Recording saved locally... Queueing sync..."`.

### Step 3: Verify Local Persistence in IndexedDB
1. In DevTools, go to **Application → Storage → IndexedDB → anc_field_db → recordings**.
2. Click **Refresh**.
3. Confirm your recording entry is present:
   - `syncState`: `"local"` or `"queued"`
   - `blob`: Binary audio blob stored
   - `createdAt`: ISO timestamp
4. Reload the web page (`F5`) while still **Offline**.
5. Navigate to the **📁 Library** tab — verify the recording persists and can be played back offline!

### Step 4: Restore Connection & Verify Sync Recovery
1. In the DevTools **Network** tab, change throttling back to **"No throttling"** (Online).
2. The network status banner will turn green: `⚡ Network Online — Auto-sync active`.
3. The Sync Engine will automatically trigger background queue processing:
   - State updates: `queued` → `uploading` → `synced`
4. Inspect the SQLite database file (`web/recordings.db`) or run GET `/api/recordings` to verify the record was received by the server.
5. Click **"🔄 Trigger Manual Sync"** again to verify **Idempotency** — the server acknowledges the existing UUID (`already_exists`) without creating duplicate records.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                   Field App (PWA)                      │
│                                                        │
│  [ Microphone ] ──> [ MediaRecorder ]                   │
│                            │                           │
│                            ▼                           │
│                  [ IndexedDB Storage ]                 │
│                 ('anc_field_db' v1)                    │
│                            │                           │
│                            ▼                           │
│                  [ Sync Engine Queue ]                 │
│              (Serial, Exponential Backoff)             │
└────────────────────────────┬───────────────────────────┘
                             │ (POST /api/recordings)
                             ▼
┌────────────────────────────────────────────────────────┐
│                 Python Backend Server                  │
│                     (web/server.py)                    │
│                            │                           │
│                            ▼                           │
│                 [ SQLite Database ]                    │
│                  (web/recordings.db)                   │
└────────────────────────────────────────────────────────┘
```

---

## Backend API Specification

| Endpoint | Method | Description | Payload / Query |
|---|---|---|---|
| `/api/status` | GET | System hardware, loaded models, CUDA / OpenVINO state | None |
| `/api/recordings` | GET | Returns list of all synced field & logbook recordings | None |
| `/api/recordings` | POST | Uploads complete field recording from client queue | `{ id, mimeType, durationSec, sampleRate, createdAt, audio_b64, sha256, prevSha256, vadSpeechSec, classification }` |
| `/api/recordings/<id>/chunk` | POST | Resumable chunked upload endpoint for Android WorkManager | `{ id, chunkIndex, totalChunks, chunk_b64, sha256, prevSha256, classification, ... }` |
| `/api/recordings/<id>/audio` | GET | Streams raw audio binary for a synced recording | `<id>` in path |
| `/api/protected_recordings` | GET | Lists top-secret duress code-word classified entries | None |
| `/api/alerts` | POST / GET | Dispatches or fetches emergency SOS alerts | `{ id, timestamp, latitude, longitude, status, details }` |
| `/api/audit_logs` | POST / GET | Ingests or fetches append-only system audit trail | `{ id, timestamp, eventType, details, sha256 }` |
| `/api/hash_chain/verify` | GET | Verifies SHA-256 Hash Chain integrity across entries | None |
| `/api/upload_and_process` | POST | Runs custom audio through ANC neural network | `{ audio_data, model_id }` |

---

## Verification Boundaries: On-Device vs. Server-Side

### 1. On-Device Native Verification (`androidTest/` & `test/` in Android)
The native Android test suite (`VoiceLoggerSystemTest.kt` & `OnnxAncProcessorTest.kt`) verifies the following behaviors on-device:
- **Zero-Gap Hourly Rotation**: Verifies segment rotation contiguity across 20s test boundaries without sample loss.
- **Service Resilience**: Verifies Foreground Service retention (`WakeLock`), auto-restart, and `BOOT_COMPLETED` receiver logic.
- **Unbroken SOS Continuity**: Verifies audio capture stream is never paused or interrupted during emergency SOS trigger.
- **Genuine On-Device KWS**: Verifies neural keyword detection (`kws_model.onnx`) returning real class labels (`"RED_TACTICAL"`, `"PHOENIX"`, `"MAYDAY"`), real float confidence scores, and zero heuristic fallback on noise.
- **ONNX ANC Signal Chain**: Verifies 512-point Hann STFT -> 3 Named ONNX Inputs (`noisy_real`, `noisy_imag`, `h_state`) -> OLA iSTFT signal chain with streaming `h_state` retention and genuine `levelReductionDb` measurement (`10*log10(in/out)`).
- **Keystore AES-256-GCM Encryption**: Verifies encryption at rest with separate primary and protected key aliases.
- **Storage Exhaustion Protection**: Verifies safe warning and stop on disk quota exhaustion.

### 2. Server-Side Verification (`scratch/verify_voice_logger.py` & `web/server.py`)
The server verification harness tests backend REST APIs and persistence:
- **SHA-256 Hash Chain Integrity**: Validates `sha256 = hash(audio || prevSha256)` across stored records via `GET /api/hash_chain/verify` and detects deliberate record tampering.
- **Resumable Chunked Sync**: Verifies 512 KB chunked upload ingestion (`POST /api/recordings/{id}/chunk`) and raw binary byte assembly.
- **Emergency SOS Alert Ingestion**: Verifies alert ingestion (`POST /api/alerts`) and retrieval (`GET /api/alerts`).
- **Append-Only System Audit Trail**: Ingests and retrieves audit events (`POST /api/audit_logs`).
- **Operator Console Dashboard**: Verifies Web Operator Dashboard (`web/static/`) styled with tactical blue/white theme.

- [x] **Sync Engine**: Serial queue, exponential backoff, resumable, idempotent UUID keys.
- [x] **Honest Telemetry & Sync States**: Displays `synced` strictly upon backend HTTP 200/201/209 acknowledgment.
- [x] **Quota Monitoring**: Checks `navigator.storage.estimate()` before saving.
