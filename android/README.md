# DRDO 24/7 Tactical Voice Logging System — Native Android Application

This directory (`android/`) contains the native Android application built in Kotlin (`minSdk 26`, `targetSdk 34`) for DRDO Problem Statement 26052.

---

## 1. Environment & Prerequisites

- **Java Development Kit (JDK)**: JDK 17 (Zulu, Temurin, or OpenJDK 17).
- **Android Studio**: Android Studio Hedgehog (2023.1.1) or newer (Iguana, Jellyfish, Koala).
- **Android Gradle Plugin (AGP)**: 8.2.0
- **Gradle Version**: 8.5 (pinned in `gradle/wrapper/gradle-wrapper.properties`).
- **Target Device**: Android 8.0 (API level 26) or higher.

---

## 2. How to Build locally

### Using Gradle Wrapper from Terminal:
```bash
# Navigate to android directory
cd android

# Build Debug APK (Linux / macOS)
./gradlew assembleDebug

# Build Debug APK (Windows PowerShell / CMD)
.\gradlew.bat assembleDebug

# Run JVM Unit Tests
.\gradlew.bat testDebugUnitTest
```

The output APK will be located at:
`android/app/build/outputs/apk/debug/app-debug.apk`

---

## 3. How to Install on a Physical Android Phone

### Option A: USB Debugging Installation
1. Open phone **Settings → About Phone**.
2. Tap **Build Number** 7 times to enable **Developer Options**.
3. Go to **Settings → System → Developer Options** and enable **USB Debugging**.
4. Connect phone via USB cable and authorize computer when prompted.
5. Install via ADB:
   ```bash
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   ```

### Option B: Sideloading CI-Produced APK
1. Download `app-debug-apk` from GitHub Actions workflow artifacts (`.github/workflows/android-build.yml`).
2. Transfer `app-debug.apk` to phone storage.
3. Open file manager on phone, tap `app-debug.apk`, and allow **"Install from unknown sources"** when prompted.

---

## 4. Required Runtime Permissions & Failure Modes

| Permission | Purpose | Impact if Denied |
|---|---|---|
| `RECORD_AUDIO` | Hardware microphone access for 24/7 capture | App fails immediately; audio stream cannot be initialized. |
| `POST_NOTIFICATIONS` | Foreground Service status banner (Android 13+) | Service fails to enter foreground state; system terminates capture. |
| `ACCESS_FINE_LOCATION` | GPS coordinates attached to Emergency SOS alerts | SOS alert dispatches with fallback coordinates `(0.0, 0.0)`. |
| `WAKE_LOCK` | CPU Partial WakeLock | Phone CPU enters deep sleep when screen turns off, suspending capture. |
| `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS` | Battery saver exemption | Android Doze Mode suspends background audio service after ~15 minutes idle. |

> [!IMPORTANT]
> To guarantee true 24/7 continuous recording, open phone **Settings → Apps → DRDO Voice Logger → Battery** and select **"Unrestricted"**.

---

## 5. Feature Flags & Known Status

### Keyword Spotting Status: DISABLED BY DEFAULT
- **Status**: The code-word feature is behind feature flag `KeywordSpotter.ENABLE_KEYWORD_SPOTTING = false`.
- **Reason**: `export/kws_model.onnx` weights are initial baseline parameters. To prevent false triggers or emitting arbitrary scores, keyword spotting is disabled until a trained model (or Picovoice Porcupine custom wake-word key) is configured.
- **SOS Flow**: The **SOS Emergency Button** (800ms press-and-hold guard) operates independently and functions fully.

---

## 6. Verification Boundaries: What is Tested & Verified

### Server-Side & API Verification (Verified via `scratch/verify_voice_logger.py`):
- [x] **SHA-256 Hash Chain Integrity**: Validates `sha256 = hash(audio || prevSha256)` across logbook entries.
- [x] **Tamper Detection**: Detects deliberate modification of stored audio payload and flags broken entry ID.
- [x] **Resumable 512 KB Chunked Uploads**: Assembles chunked uploads (`POST /api/recordings/{id}/chunk`).
- [x] **Emergency SOS Alert Ingestion**: Dispatches and lists SOS alerts (`POST /api/alerts`).
- [x] **System Audit Trail**: Ingests and stores append-only audit events (`POST /api/audit_logs`).

### On-Device Hardware Verification (Requires physical Android device / emulator):
- [ ] **Physical Microphone AudioRecord**: Requires physical mic stream on device.
- [ ] **Foreground Service Doze Survival**: Requires running 24+ hours on physical hardware under Android Doze mode.
- [ ] **Instrumented Android Tests**: Execute `.\gradlew.bat connectedCheck` with a connected Android device or running emulator to run native UI and hardware tests.
