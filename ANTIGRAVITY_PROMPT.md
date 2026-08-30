# Antigravity prompt — make the Android app actually build and install

Copy everything between the horizontal rules into Antigravity, with the repo open.

---

The Android app in `android/` has **never been compiled**. Not once. Verified on this machine:

```
APK              : does not exist
Gradle wrapper   : missing  (no gradlew / gradlew.bat / gradle-wrapper.jar)
Android SDK      : not installed
Java / JDK       : not installed
local.properties : missing
```

Consequently the "instrumented test suite" reported earlier as covering zero-gap rotation,
service resilience and SOS continuity **has never executed** — there is no JDK to run it with.
Do not report those tests as passing again until they have actually run and you can paste the
output.

**This task has one goal: a working APK installed on a physical Android phone.** UI polish comes
after that, because the UI cannot be judged until it can be seen on a device.

## Step 1 — make the project buildable

1. **Add the Gradle wrapper**: `gradlew`, `gradlew.bat`, and
   `gradle/wrapper/gradle-wrapper.jar` + `gradle-wrapper.properties`. Pin a Gradle version
   compatible with the Android Gradle Plugin in `android/build.gradle`. Without the wrapper no
   build is possible at all.
2. **Add `android/local.properties`** with `sdk.dir` (and **gitignore it** — it is machine-local).
3. **Audit every dependency version** in `app/build.gradle` for mutual compatibility: AGP,
   Kotlin, Room, WorkManager, ONNX Runtime Mobile, Material 3. Version skew is the most common
   cause of a first build failing.
4. **Fix the ONNX asset problem.** `app/src/main/assets/kws_model.onnx` stores its weights
   externally in `kws_model.onnx.data`, which is **not in assets/** — it is in `export/`. The
   model therefore cannot load on device. Either bundle the `.data` file alongside it or
   re-export the model with weights embedded in a single file.
5. **Verify `causal_anc_model.onnx` (3.2 MB) is correctly packaged** and loadable from assets.

## Step 2 — make it compile, and prove it

Run a real build and paste the actual output:

```
cd android && ./gradlew assembleDebug
```

Fix every compilation error until it succeeds. Then run the unit tests:

```
./gradlew testDebugUnitTest
```

**Report the true result.** If tests fail, say which and why. A failing test reported honestly
is worth more than a green summary that never ran.

## Step 3 — CI fallback build

Add `.github/workflows/android-build.yml` that builds `assembleDebug` on push and uploads the
APK as a workflow artifact. This gives a working build path even on a machine with no local
Android SDK — this laptop has **Smart App Control enforcing**
(`VerifiedAndReputablePolicyState=1`), which has already blocked one unsigned installer, so a
cloud build route is genuinely useful insurance.

## Step 4 — installation instructions

Write `android/README.md` covering, concretely:
- Required JDK version and Android Studio version
- How to open, sync and build the project
- How to enable Developer Options + USB debugging and install to a physical phone
- How to sideload the CI-produced APK instead
- Which runtime permissions must be granted (microphone, notifications, location, battery
  optimisation exemption) and what breaks if each is denied
- **What is verified and what is not** — be explicit that on-device behaviour is unverified
  until the instrumented tests have actually run on hardware

## Known-broken, do not paper over

**The keyword spotter model is untrained.** `export/kws_model.onnx` weights are raw PyTorch
default initialisation — `fc.0.weight` bounded at ±0.0624 = `1/sqrt(257)`, `fc.2.weight` at
±0.125 = `1/sqrt(64)`. No training script and no training data exist for these code words.
Structurally it is also wrong for the task: input is `input_features (1, 257)`, a **single STFT
frame**, while a spoken keyword is a temporal pattern spanning roughly a second — a per-frame
MLP cannot represent it regardless of training.

Therefore: **disable the code-word feature behind a flag defaulting to OFF**, and label it in
the UI and README as "not implemented — requires a trained keyword model". Do **not** ship a
random-weight classifier that emits a plausible-looking softmax confidence. It is worse than an
obvious heuristic because it survives code review while behaving as noise.

When it is implemented for real, use **Picovoice Porcupine** with a custom wake word, AccessKey
in gitignored `local.properties`.

## Non-negotiables

- **Never report a test as passing unless you ran it and can paste the output.**
- **Never display a metric the app did not compute.** This repository has a documented history
  of fabricated numbers — a fake PESQ, a fake NPU latency, a false TensorRT label, and a
  hardcoded 15.26 dB. Unavailable means "not measured".
- Do not touch `model/`, `data/`, `eval/`, `scripts/`, `checkpoints/`, `demo/` or `benchmark/`.
- `web/server.py` is Python **stdlib `http.server`** — do not swap it for Flask or FastAPI.

## Deliverable

A `.apk` file that installs and runs on a physical Android phone, plus the pasted output of a
successful `./gradlew assembleDebug`. Nothing else counts as done.
