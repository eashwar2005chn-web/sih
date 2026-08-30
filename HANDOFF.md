# Handoff — DRDO PS 26052 Causal ANC

Paste the block under **"Prompt to give Claude Code"** into a fresh Claude Code session opened
in the project root. Do the setup first or the training scripts will fail loudly.

---

## Setup (do this first, ~15 min)

**1. The 100-hour training corpus is NOT in this zip.** It lives outside the project directory
and is 6.3 GB. Download it fresh — it is a public corpus:

```bash
curl -L -o train-clean-100.tar.gz https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz
```

Then point the loader at it. Default location is `~/datasets_local/LibriSpeech/train-clean-100`;
anywhere else needs the environment variable:

```bash
export LIBRISPEECH_TRAIN100_DIR=/your/path/LibriSpeech/train-clean-100
```

Verify (all three must PASS, and it must report 100.59 h):

```bash
python scripts/verify_disjoint_training.py
```

**2. Python deps.** Needs `torch` (CUDA build), `numpy`, `scipy`, `soundfile`, `pystoi`,
`sounddevice`, `openvino` (optional), `streamlit` + `plotly` (demo only). Python 3.11.

**3. Confirm the evaluation splits are intact** — this guards every published number:

```bash
python scripts/freeze_eval_splits.py
```

---

## Prompt to give Claude Code

> I'm continuing work on a DRDO PS 26052 causal adaptive noise cancellation project. Read
> `eval/PS_REQUIREMENTS_COMPLIANCE.md`, `eval/SPLIT_INTEGRITY_AND_DATA_REGIME.md`, and
> `eval/experiments_planned.csv` first — the CSV is the experiment ledger and is the source of
> truth for what has been tried and what the results were. `eval/DRDO_TARGETS_EVALUATION.md` is
> the source of truth for all published metrics.
>
> **State of record:** the model of record is `checkpoints/phase22_LONG150K/best_model.pt` —
> 15.26 dB test SNR [14.55, 15.97], 44.3% per-utterance pass rate, STOI 0.9309, PESQ 3.96 (an
> uncalibrated proxy, not real ITU-T P.862). 811,522 params, 3.1 MB, RTF 0.0085.
>
> **A retrain was in progress when this was handed over.** `scripts/run_phase24_disjoint_100h.py`
> trains the same architecture on a 100.59 h speaker-disjoint corpus instead of the original
> 5.04 h corpus that shared both speakers and 30% of its content with the test set. It writes
> full optimizer + RNG state every epoch, so just run it again and it resumes from the last
> completed epoch:
>
> ```
> python scripts/run_phase24_disjoint_100h.py
> ```
>
> Its control is P22-LONG at 15.00 dB val. Note the control was measured WITH speaker and
> content overlap while this run has neither, so a smaller delta can still be the better model.
> The result may legitimately come in below 15.26 dB.
>
> **These rules are not negotiable and matter more than any result:**
> - Never fabricate or estimate a metric. Every number must trace to a script and an artifact
>   you can point at. If something cannot be measured, say so — the repo has a history of
>   fabricated numbers that were caught and documented, and that must not recur.
> - All model-selection decisions are made on the VALIDATION split. The test set is evaluated
>   at most once per finalized model.
> - Never change the val/test file lists. `verify_eval_splits_frozen()` runs at the top of every
>   evaluation script and raises if they drift. If it fires, stop and investigate — do not
>   re-freeze.
> - A new training corpus must be pinned to train via `PINNED_SOURCES` in
>   `data/real_dataset_loader.py`, otherwise the path-hash splitter leaks ~20% of it into
>   val/test and silently invalidates every prior number.
> - Retrain budget: 6 of 8 used. Phase 24 is number 6.
>
> **Tasks, in priority order:**
> 1. Let Phase 24 finish (75 epochs, ~1 min/epoch on an RTX 5060). Report the result against the
>    control honestly, including if it lost.
> 2. If and only if Phase 24 is the better model, run a SINGLE test-set evaluation with
>    `scripts/evaluate_phase22_final.py` as the template, then propagate one consolidated set of
>    numbers to `README.md`, `docs/JUDGE_PITCH_ONEPAGER.md`, `docs/COMPETITION_NARRATIVE.md`, and
>    `docs/DEMO_VIDEO_SCRIPT.md`. Those docs are deliberately stale right now — do not update
>    them piecemeal.
> 3. Train the sub-band model (`model/causal_anc_subband.py`, PS requirement B2). The
>    architecture is implemented and gated — it adds 1.5% params, zero extra lookahead, RTF
>    0.0223 — but has never been trained. Use the Phase 24 script as the template, changing only
>    the model class. That is retrain 7 of 8.
> 4. Investigate the unexplained training-loss transients at Phase 24 epochs 10 and 24 (loss
>    jumped to 22.4 and 14.3 for one epoch, then returned to trend). The control never exceeded
>    0.499 after epoch 5. Two hypotheses are already refuted and recorded on the `P24-DISJOINT`
>    ledger row: near-silent clean files, and random windows landing in silence. Impact was
>    assessed as nil — val stayed on trend, zero non-finite skips, gradient clipping bounds the
>    update — but the cause is still unknown. Instrument per-batch loss to find it.
>
> **Known blockers — do not burn time on these, they are environmental:**
> - Real ITU-T P.862 PESQ could not be installed on the original machine (Smart App Control
>   enforcing, no compiler, no prebuilt wheel). If your machine allows it, try
>   `conda install -c conda-forge pesq` or `pip install pesq` with MSVC build tools present —
>   this would be genuinely valuable, since PESQ is currently the weakest claim in the
>   submission. Recompute with `eval/metrics_common.py` and update
>   `DRDO_TARGETS_EVALUATION.md` §4.1 if it works.
> - TensorRT and Jetson AGX Orin deployment are unimplemented for lack of hardware/runtime, and
>   are documented as such.

---

## Useful commands

| | |
|---|---|
| Resume/continue training | `python scripts/run_phase24_disjoint_100h.py` |
| Verify splits unchanged | `python scripts/freeze_eval_splits.py` |
| Verify disjoint regime | `python scripts/verify_disjoint_training.py` |
| Sub-band gate (causality + latency) | `python scripts/verify_subband_causality_latency.py` |
| Pruning sweep | `python scripts/prune_model.py` |
| Live mic demo (no mic needed) | `python demo/live_anc.py --selftest` |
| Live mic demo (real) | `python demo/live_anc.py --duration 30` |
| Dashboard | `streamlit run demo/app.py` |

## What can be deleted before zipping (~3 GB)

Already-extracted archives and superseded checkpoints:

```bash
rm datasets/dev-clean.tar.gz datasets/esc50.zip
rm -rf checkpoints/phase16_* checkpoints/phase18_* checkpoints/phase19_* checkpoints/phase2[01]_*
```

Keep `checkpoints/phase22_LONG150K/` (model of record), `checkpoints/phase24_disjoint_100h/`
(in-progress + resume state), `checkpoints/task4_full_run/` (published baseline), and all of
`datasets/clean_speech`, `datasets/generic_noise`, `datasets/tactical_noise`, `datasets/rirs`.
