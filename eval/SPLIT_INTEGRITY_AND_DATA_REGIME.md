# Split integrity and the training-data regime

**Date: 2026-08-30 · No retrain consumed · Budget remains 5 of 8**

Three defects in how data reaches the model were found, disclosed, and fixed at the source.
None of them changes a published number; all three change what a future number *means*.

---

## 1. What was wrong

### 1.1 The test split was not speaker-disjoint

`RealAudioManifest.get_split_files()` assigns files by `md5(filepath) % 100`. That guarantees
no *file* is shared between splits — but a speaker's utterances scatter across all three.

**Measured: 41 of 41 test speakers (100%) were also in training. Zero unseen speakers.**

Every published test figure, the 15.26 dB headline included, therefore measures *same-speaker*
performance. For a speech enhancement system the operationally relevant question is the
opposite one: how it handles a voice it has never heard.

### 1.2 The `voicebank` source is relabelled dev-clean

`datasets/clean_speech/voicebank` is **not VoiceBank**. Its filenames embed LibriSpeech
utterance ids — `p225_000_`**`1272-128104-0000`**`.wav` — and the decoded waveforms are
identical to the corresponding `dev-clean` files (**correlation 1.0000**).

So the same audio is indexed under two sources, and the path hash sends the two copies to
different splits. This is the mechanism behind the content leak Phase 15 measured:
**103 of 342 test clean-speech files (30.12%)** shared content with training.

Phase 15 found this, quantified it, showed it does **not** inflate any metric (every CI spans
zero; the model performs *worse* on leaked items in the stress buckets), and wrote that it
"must be disclosed." It had not reached the main evaluation report. It is now in
`DRDO_TARGETS_EVALUATION.md` §4.4.

### 1.3 Adding data would have silently redefined the test set

The path hash treats any new corpus the same way: drop in 28,539 files and ~10% land in val,
~10% in test. Measured on a synthetic stand-in: **209 files into val, 203 into test.** Every
prior number would have become incomparable while still looking valid.

### 1.4 The data pipeline was not in version control

`.gitignore` listed `data/` under the heading *"Large Raw Audio Datasets"*. But `data/` holds
three source modules and no audio (~160 KB); the audio is under `datasets/`. The entire data
pipeline — split logic, mixing, the `worker_init_fn` RNG fix — was untracked, so a fresh clone
could not run. Fixed; `datasets/` remains ignored.

---

## 2. What was built

**Split pinning.** `PINNED_SOURCES` assigns a source wholesale to one split, bypassing the
hash. A pinned source contributes all of its files to that split and none to any other.

**A frozen evaluation manifest.** `eval/frozen_eval_manifest.json` holds a sha256 over the
sorted val/test file lists per category, captured at **val=666 / test=665 files before any new
data was present**. `verify_eval_splits_frozen()` raises on drift and now runs at the top of
`evaluate_phase10_final`, `evaluate_phase17_final`, `evaluate_phase22_final`, and
`eval/evaluate_drdo_targets`.

**One explicit regime switch.** `set_speaker_disjoint_training(bool)`:

| | default (`False`) | disjoint (`True`) |
|---|---|---|
| Training corpus | dev-clean derived, **5.04 h** | train-clean-100, **100.59 h** |
| Train speakers | 41 | 251 |
| Speaker overlap with test | **41 of 41** | **0** |
| Content overlap with test | **30.12%** | **0.00%** |
| val / test | 346 / 342 files | **identical** |

Default is `False` so every published run reproduces byte-for-byte. Enabling it raises if the
corpus is absent, rather than quietly training on almost nothing.

---

## 3. Verification

Both guards are tested **with a negative control**, because a guard that never fires would
also pass:

`scripts/test_split_pinning_guard.py` — 2,000 synthetic clean-speech files:

```
[1] PINNED to train        val/test sha256 unchanged ........... YES
[2] NEGATIVE CONTROL       same files unpinned:
                           209 leaked into val, 203 into test
                           guard detects drift ................. YES
```

`scripts/verify_disjoint_training.py` — the real 28,539-file corpus, all three properties at
once:

```
[1] val/test byte-identical to frozen manifest ......... PASS
[2] speaker overlap train x test ....................... PASS  (train 251, test 41, shared 0)
[3] content overlap train x test ....................... PASS  (0 of 336 test clips = 0.00%)

training corpus :  28539 clean-speech files   100.59 h
val / test      :    346 /    342 files (unchanged)
```

Corpus provenance: `openslr.org/resources/12/train-clean-100.tar.gz`, 6,387,309,499 bytes,
gzip integrity verified, 251 speaker directories. Stored outside the repo and outside OneDrive
at `~/datasets_local/`; override with `LIBRISPEECH_TRAIN100_DIR`.

---

## 4. Why this is the next retrain

After Phases 16–23, four candidate constraints are eliminated: mask shape, optimizer steps,
supervision density, and capacity. Phase 23 was decisive on the last — a 4× model reached the
**same final train loss** (−0.2370 vs −0.2383) and slightly lower validation. Training-data
volume is the only constraint still standing, and the prior corpus was LibriSpeech
*dev-clean* — the **development** split — at 5.04 h, roughly 9,073 distinct 2-second segments
each reused ~225 times across 150,000 steps. The field norm is 100–500 h.

The same change also repairs §1.1 and §1.2, because `train-clean-100` is speaker-disjoint from
`dev-clean` by corpus construction.

### The honest risk

**The result may come in below 15.26 dB.** More data pushes up; removing same-speaker and
same-content overlap removes an optimistic bias and pushes down. Which dominates is not
knowable in advance.

That is the correct trade to make. A number measured on unseen speakers with no content
overlap is a number that survives scrutiny; the current one is one careful question away from
being discounted entirely.

**Control:** P22-LONG at 15.00 dB val. **Recipe:** identical — 150,000 steps, fp32, warmup 500
→ peak 5e-4 → cosine 1e-5, batch 16, seed 42, workers 8 — so training data is the only
variable. Would be **retrain 6 of 8**. Not started.
