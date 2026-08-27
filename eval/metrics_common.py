"""
Consolidated Standardized Metric Engine (DRDO PS 26052)
Single source of truth for SNR, STOI, PESQ (approx.), Wilson Score CIs, and standard errors.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional
from model.loss import compute_snr, compute_stoi, compute_pesq


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score interval for binomial proportion (as percentages)."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.96 if confidence == 0.95 else 2.576
    p_hat = successes / total
    denominator = 1 + (z**2) / total
    centre_adj = p_hat + (z**2) / (2 * total)
    spread = z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * total)) / total)
    lower = max(0.0, (centre_adj - spread) / denominator)
    upper = min(1.0, (centre_adj + spread) / denominator)
    return (float(lower * 100.0), float(upper * 100.0))


def evaluate_batch_metrics(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics for a single audio sample (1D numpy arrays)."""
    raw_snr = compute_snr(clean, noisy)
    enh_snr = compute_snr(clean, enhanced)
    snr_gain = enh_snr - raw_snr
    raw_stoi = compute_stoi(clean, noisy)
    enh_stoi = compute_stoi(clean, enhanced)
    raw_pesq = compute_pesq(clean, noisy)
    enh_pesq = compute_pesq(clean, enhanced)

    return {
        "raw_snr": float(raw_snr),
        "enh_snr": float(enh_snr),
        "snr_gain": float(snr_gain),
        "raw_stoi": float(raw_stoi),
        "enh_stoi": float(enh_stoi),
        "raw_pesq": float(raw_pesq),
        "enh_pesq": float(enh_pesq),
    }


def aggregate_metric_records(records: List[Dict[str, float]]) -> Dict[str, Union[float, Tuple[float, float], int]]:
    """
    Standardized aggregation across a list of per-sample metric dicts.
    Computes mean, std, SEM, 95% CIs, and target pass rates with Wilson score intervals.
    """
    if not records:
        return {}

    n = len(records)
    in_snrs = np.array([r["raw_snr"] for r in records])
    out_snrs = np.array([r["enh_snr"] for r in records])
    gains = np.array([r["snr_gain"] for r in records])
    stois = np.array([r["enh_stoi"] for r in records])
    pesqs = np.array([r["enh_pesq"] for r in records])

    # Means, standard deviations, standard errors
    mean_in = float(np.mean(in_snrs))
    mean_out = float(np.mean(out_snrs))
    std_out = float(np.std(out_snrs, ddof=1)) if n > 1 else 0.0
    sem_out = std_out / np.sqrt(n) if n > 1 else 0.0
    ci_out_95 = (mean_out - 1.96 * sem_out, mean_out + 1.96 * sem_out)

    mean_gain = float(np.mean(gains))
    std_gain = float(np.std(gains, ddof=1)) if n > 1 else 0.0

    mean_stoi = float(np.mean(stois))
    std_stoi = float(np.std(stois, ddof=1)) if n > 1 else 0.0
    sem_stoi = std_stoi / np.sqrt(n) if n > 1 else 0.0
    ci_stoi_95 = (mean_stoi - 1.96 * sem_stoi, mean_stoi + 1.96 * sem_stoi)

    mean_pesq = float(np.mean(pesqs))
    std_pesq = float(np.std(pesqs, ddof=1)) if n > 1 else 0.0
    sem_pesq = std_pesq / np.sqrt(n) if n > 1 else 0.0
    ci_pesq_95 = (mean_pesq - 1.96 * sem_pesq, mean_pesq + 1.96 * sem_pesq)

    # Pass rates against DRDO targets
    pass_snr_count = int(np.sum(out_snrs >= 15.0))
    pass_snr_pct = (pass_snr_count / n) * 100.0
    pass_snr_ci = wilson_score_interval(pass_snr_count, n, 0.95)

    pass_stoi_count = int(np.sum(stois >= 0.850))
    pass_stoi_pct = (pass_stoi_count / n) * 100.0
    pass_stoi_ci = wilson_score_interval(pass_stoi_count, n, 0.95)

    pass_pesq_count = int(np.sum(pesqs >= 2.50))
    pass_pesq_pct = (pass_pesq_count / n) * 100.0
    pass_pesq_ci = wilson_score_interval(pass_pesq_count, n, 0.95)

    return {
        "n": n,
        "mean_in_snr": mean_in,
        "mean_out_snr": mean_out,
        "std_out_snr": std_out,
        "sem_out_snr": sem_out,
        "ci_out_snr_95": ci_out_95,
        "mean_gain": mean_gain,
        "std_gain": std_gain,
        "pass_snr_count": pass_snr_count,
        "pass_snr_pct": pass_snr_pct,
        "pass_snr_ci": pass_snr_ci,
        "mean_stoi": mean_stoi,
        "std_stoi": std_stoi,
        "sem_stoi": sem_stoi,
        "ci_stoi_95": ci_stoi_95,
        "pass_stoi_count": pass_stoi_count,
        "pass_stoi_pct": pass_stoi_pct,
        "pass_stoi_ci": pass_stoi_ci,
        "mean_pesq": mean_pesq,
        "std_pesq": std_pesq,
        "sem_pesq": sem_pesq,
        "ci_pesq_95": ci_pesq_95,
        "pass_pesq_count": pass_pesq_count,
        "pass_pesq_pct": pass_pesq_pct,
        "pass_pesq_ci": pass_pesq_ci,
    }
