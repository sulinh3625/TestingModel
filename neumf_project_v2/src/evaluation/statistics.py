from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "n": int(len(arr))}


def paired_wilcoxon(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Hai dãy phải có cùng số seed.")
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0}
    stat, p = wilcoxon(a, b)
    return {"statistic": float(stat), "p_value": float(p)}
