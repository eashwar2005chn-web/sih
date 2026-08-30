# Phase 23 — Fair capacity test: 3.24M vs 811K at an identical 150,000-step budget

**Status: NEGATIVE. Not promoted. No test-set evaluation performed.**
**Date: 2026-08-30 · Ledger row `P23-CAP` · Retrain 5 of 8**

---

## 1. What this tested and why it needed testing

Phases 5–7 concluded "capacity is ruled out." That conclusion was drawn at **3,750 optimizer
steps**, a budget at which no model of any size could have demonstrated a benefit — a larger
model needs more steps before its extra parameters do anything, so the experiment could only
ever have returned "no difference." The verdict was therefore untrustworthy, not wrong.

Phase 22 (P22-LONG) then finished with **train loss at −0.24 while validation oscillated
within ±0.02 dB** across its final ten epochs. I read that signature as a capacity limit and
recorded it as such. This phase was the fair re-test of that reading, at a budget where the
811K model demonstrably still improves.

## 2. Design

Capacity is the **only** variable. Everything else is byte-identical to P22-LONG.

| | Control (P22-LONG) | Treatment (P23-CAP) |
|---|---|---|
| Architecture | `CausalANCNet` hidden 128 | `CausalANCScaledNet` hidden 256 |
| Parameters | 811,522 | **3,235,842 (4.0×)** |
| Steps | 150,000 (75 ep × 32,000 ÷ 16) | 150,000 (identical) |
| Precision | fp32 | fp32 |
| LR schedule | warmup 500 → peak 5e-4 → cosine 1e-5 | identical |
| Batch / seed / workers | 16 / 42 / 8 (RNG fix active) | identical |
| Loss | `HybridANCLoss(clamped_alpha)` | identical |
| Selection split | validation | validation |

Promote rule, fixed before the run: **> +1.0 dB val over 15.00 dB**, or no test evaluation.

## 3. Result

```
  3.24M model   :  +14.94 dB   STOI 0.9161   (best, epoch 70)
  811K control  :  +15.00 dB   STOI 0.9175   (best, epoch 73)
  delta         :   -0.06 dB   ->  DO NOT PROMOTE
```

The test set was **not touched**. Per the Section D decision rule, a treatment that loses on
validation does not get a test evaluation — evaluating it anyway would be test-set shopping.

### Epoch-matched curves

| Epoch | 811K val | 811K train | 3.24M val | 3.24M train |
|---:|---:|---:|---:|---:|
| 5 | +13.35 | +0.5858 | +13.18 | +0.7112 |
| 10 | +14.00 | +0.2852 | +13.72 | +0.3665 |
| 15 | +14.26 | +0.1665 | +14.13 | +0.2369 |
| 25 | +14.52 | +0.0225 | +14.35 | +0.0758 |
| 30 | +14.57 | −0.0454 | +14.48 | +0.0011 |
| 50 | +14.84 | −0.1749 | +14.77 | −0.1611 |
| 70 | **+14.99** | −0.2476 | **+14.94** | −0.2413 |
| 75 | +14.98 | −0.2383 | +14.89 | **−0.2370** |

The 4× model was at or behind the control at **every** sampled epoch. There is no crossover
and no late divergence.

## 4. Interpretation — the decisive number is the train loss, not the val

The two runs converge to **essentially the same final training loss: −0.2370 (3.24M) vs
−0.2383 (811K)**, a gap of 0.0013.

This is what settles it. A model that is capacity-limited, handed 4× the parameters, fits the
**training** data materially better — that is close to the definition of the term. This one
did not fit it better at all. So the residual loss is not something more parameters can
absorb.

**This refutes the Phase 22 inference, which was mine.** "Train loss falling while val is
pinned" is consistent with a capacity limit, but it is equally consistent with the model
having already extracted everything the input determines, with the remaining train-loss
movement being fit to sample-specific noise that does not transfer. The two readings are
distinguished only by the experiment just run, and it came back against my reading. The
P22-LONG ledger row has been amended to record this, with the original wording retained for
provenance.

Note also what did **not** happen: the larger model did not overfit — it never opened a
train-loss lead to overfit from. It simply tracked the smaller model slightly worse
throughout, consistent with a harder optimisation problem at equal steps and no additional
signal available to exploit.

### What the constraint actually is

By elimination across the ledger, the ~15 dB plateau is not:

- **not architecture / mask shape** — the head can represent 27.57 dB (`probe_representational_vs_estimation.py`); linear and leaky clamps both lost to tanh (C1-S1/S2)
- **not optimizer steps** — 3,750 → 30,000 bought +1.60 dB; 30,000 → 150,000 bought +0.83 dB; saturating
- **not supervision density** — closed-form per-bin cRM supervision held parity, +0.09 dB (P21-B)
- **not capacity** — this phase, −0.06 dB at 4× parameters and equal train loss

What remains is **the data and the task**: training-set diversity, and the irreducible
ambiguity of recovering a clean target from a single noisy causal observation with ≈16 ms of
lookahead. The oracle ceiling of 29.46 dB is computed *with access to the clean signal*, so it
bounds what is representable, not what is estimable from the input alone.

## 5. Budget and state of record

**Retrains used: 5 of 8** (C7-L1, P21-A, P21-B, P22-LONG, P23-CAP). Bug-invalidated runs
excluded per the user ruling of 2026-08-30.

**Model of record remains P22-LONG**, unchanged by this phase:

| Metric | Value | Target | Verdict |
|---|---|---|---|
| Operational SNR (test, N=500) | **15.26 dB** [14.55, 15.97] | > 15 dB | **MET** (point estimate; CI straddles) |
| Per-utterance pass rate | 44.3% | 80% (self-imposed, not in the PS) | NOT MET |
| STOI | 0.9309 | > 0.85 | MET |
| PESQ (uncalibrated Bark proxy) | 3.96 | > 2.5 | MET, **proxy not ITU-T P.862** |

Standing caveats on that row are unchanged: the CI straddles 15.0; the +0.91 dB over Phase 17
is not statistically distinguishable (CI [−0.06, +1.88]); and a winner's-curse discount applies
because two models have now been scored on the same fixed test set.
