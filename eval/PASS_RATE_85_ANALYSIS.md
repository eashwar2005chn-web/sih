# Reaching an 85% per-utterance pass rate — what it would actually take

**Date: 2026-08-30 · All figures recomputed from the committed per-utterance CSV of the model of record**
**Sources: `scripts/pass_rate_feasibility.py`, `scripts/variance_decomposition_probe.py`**

Current state, operational buckets, N=300, from
`eval/test_eval_csvs/Phase22_Long150k_811K_test_eval.csv`:

```
mean 15.26 dB    sd 6.30    pass 44.3%  [38.8, 50.0]
```

---

## 1. The finding that reframes the problem

A pass rate is a function of **two** quantities — the mean *and* the spread. Every experiment
in the ledger across Phases 5–23 targeted the mean. **None targeted the spread.**

Required mean to hit each target, as a function of sd:

| sd | for 80% | for 85% |
|---:|---:|---:|
| **6.30 (today)** | 20.30 dB (**+5.04**) | 21.53 dB (**+6.27**) |
| 5.00 | 19.21 dB (+3.95) | 20.18 dB (+4.92) |
| 4.00 | 18.37 dB (+3.11) | 19.15 dB (+3.88) |
| 3.00 | 17.52 dB (+2.26) | 18.11 dB (+2.85) |
| 2.50 | 17.10 dB (+1.84) | 17.59 dB (+2.33) |
| 2.00 | 16.68 dB (+1.42) | 17.07 dB (+1.81) |

At today's spread, 85% needs **+6.27 dB** — larger than every gain the project has ever
produced, combined. At sd 2.5 it needs **+2.33 dB**, which is in the range of a single
successful intervention.

Distribution-free cross-check (shift every utterance by a constant, no normality assumed):
**+4.83 dB** for 85%, **+4.11 dB** for 80%.

## 2. Where the failures actually are

| bucket | N | mean | sd | pass | fails | share of all failures |
|---|---:|---:|---:|---:|---:|---:|
| Operational [0, 5) dB | 100 | 12.10 | 6.60 | **16.0%** | 84 | **50.3%** |
| Operational [5, 10) dB | 100 | 16.00 | 6.07 | 46.0% | 54 | 32.3% |
| Operational [10, 15] dB | 100 | 17.68 | 4.78 | 71.0% | 29 | 17.4% |

Two things to take from this:

- **Half of all failures live in [0, 5) dB.** That bucket passes 16%.
- **Even the easiest bucket only passes 71%.** For 85% overall, [10,15] would have to reach
  ~95% *and* the hard buckets improve substantially. That is a demanding joint requirement.

How close are the failures?

```
shortfall below 15 dB:  median 2.87   75th 4.93   90th 7.26   max 12.56
within 1 dB of passing:  26 utterances (15.6% of failures)
within 3 dB of passing:  88 utterances (52.7% of failures)
```

**Over half of failures are within 3 dB.** The distribution is not hopeless — it is clustered
just under the line.

## 3. The largest measured opportunity: gain calibration

`scripts/variance_decomposition_probe.py`, validation split, N=300:

```
BASE   model output                      mean +15.40   sd 5.41   pass 45.7%
GAIN   + oracle per-utterance rescale    mean +16.66   sd 5.92   pass 59.7%
```

**+14.0 percentage points of pass rate from a scalar per utterance.** Per bucket:

| bucket | base pass | gain-fixed pass |
|---|---:|---:|
| Op [0,5) | 26.0% | 38.0% |
| Op [5,10) | 44.0% | 56.0% |
| Op [10,15] | 67.0% | **85.0%** |

### Why this gap exists

The training objective is **SI-SNR — scale-*invariant***. The evaluation metric is plain SNR —
scale-*dependent*. The model is therefore **never asked to get absolute gain right**, and it
doesn't: the optimal per-utterance rescale has median α = 0.9912, **sd 0.0667**, 5–95% range
[0.844, 1.012].

Because α varies per utterance, **a single global constant cannot remove it** — this was
measured directly in an earlier probe (global rescale recovered only +0.06 dB against +1.06 dB
for per-utterance). A deployable fix has to *estimate* α, e.g. an auxiliary scalar head
supervised against the closed-form optimum, or a scale-dependent term in the loss.

**Honest caveats.** The +14.0 pp figure uses an **oracle** α computed from the clean reference;
a learned estimator will recover less. And the fix *widens* the spread slightly (sd +0.52), so
it is a mean-shift that converts near-threshold utterances, not variance reduction. I described
it as a variance lever earlier in this investigation; that was wrong.

## 4. Every remaining avenue, ranked

### Raising the mean

| | Status | Notes |
|---|---|---|
| **Data 5 h → 100 h** | **Running (Phase 24)** | Only constraint not eliminated by Phases 16–23 |
| Subband processing (C.6) | Untested | PS explicitly asks for full-band *and* sub-band features |
| Target-speaker extraction | Untested, scope change | Adds information (enrollment clip) rather than reorganising it |

Exhausted, with evidence: mask shape (C1, both variants lost), optimizer steps (saturating:
+1.60 dB then +0.83 dB at 5× compute), supervision density (P21-B, +0.09 dB), capacity
(P23, −0.06 dB at 4× params with **identical final train loss**), VAD gating (+0.07 dB with a
perfect oracle detector, because only 0.183% of output energy sits in speech-absent frames).

### Shrinking the spread — the unexploited axis

| | Status | Evidence |
|---|---|---|
| **Learned gain calibration** | **Untested** | **+14.0 pp with oracle α (§3)** |
| Hard-example weighting | Genuinely untested | C2-B attempted it but was invalidated by the duplicate-data bug |
| Threshold-aware loss | Untested | Currently optimising mean SNR while the metric is P(SNR > 15) — a hinge or quantile loss concentrates gradient near the line |

The threshold-aware loss carries a trade-off that must be disclosed if used: it can raise the
pass rate **while lowering** mean SNR, by conceding hopeless utterances to push marginal ones
over. That is arguably what an operational requirement asks for, but it must be reported as
what it is.

## 5. Off the table

These would move the number without improving the system, and none will be used:

- Reporting only the [10, 15] dB bucket, or dropping the stress buckets from any headline
- Redefining the metric to SI-SNR, which reads higher than the scale-dependent SNR in use
- Excluding hard utterances, or re-drawing the test set
- Re-freezing the evaluation manifest to obtain a more favourable sample

## 6. Honest verdict

**85% is unlikely to be reachable, and I would not plan around it.**

The requirement is **+4.83 dB uniform** at the current spread. The single largest measured
opportunity — gain calibration — is worth **+14.0 pp under an oracle**, and a learned version
will deliver less. Data, subband, and hard-example weighting are each plausibly worth a
decibel or so, unconfirmed. Stacking every one of them optimistically lands nearer **65–70%**
than 85%.

What is genuinely available:

1. **Phase 24** (running) — data, and it simultaneously removes the speaker/content overlap
2. **Gain calibration** — the biggest measured pp opportunity in the project, untested, one retrain
3. **Threshold-aware loss** — directly targets the actual metric, one retrain

That is 3 of the 3 remaining retrains, and a realistic ceiling in the 60s.

**Worth restating:** the 80% and 85% pass-rate figures are **self-imposed internal targets**.
The problem statement asks for **SNR > 15 dB, STOI > 0.85, PESQ > 2.5** — all three of which
the current model meets on point estimates. A defensible 60% pass rate with a clean
methodology is a stronger submission than an 85% that depends on a metric nobody asked for.
