"""Statystyki konfiguracji wg Instrukcja-test-podstawa, Krok 3-5.

f_r - najlepsza wartosc funkcji celu z r-tego uruchomienia, r = 1..R.
  f_min, f_max, srednia, mediana,
  s_f = sqrt( 1/(R-1) * sum (f_r - srednia)^2 )   (poprawka R-1),
  CV  = s_f / |srednia| * 100 %,
  gap wzgledem f* : (x - f*) / f* * 100 %.

Optimum instancji glownej jest nieznane, wiec f* to najlepsze znane f (minimum ze wszystkich uruchomien
w analizowanym zbiorze). Dla instancji "tiny" f* to prawdziwe optimum z brute-force.
CV ma tu sens, bo f > 0 i srednia jest wyraznie rozna od zera (inaczej niz dla benchmarkow z optimum 0).
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from project.tester.runner import exact_optimum

GROUP_KEYS = ("algo", "instance", "set_id", "N", "I")


def describe_values(values: list[float] | np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    return {
        "R": len(v),
        "f_min": float(v.min()),
        "f_max": float(v.max()),
        "f_mean": mean,
        "f_median": float(np.median(v)),
        "f_std": std,
        "f_cv_pct": std / abs(mean) * 100.0 if mean != 0 else float("nan"),
        "f_q1": float(np.percentile(v, 25)),
        "f_q3": float(np.percentile(v, 75)),
    }


def reference_values(records: list[dict]) -> dict[str, tuple[float, bool]]:
    """f* dla kazdej instancji: (wartosc, czy_dokladna). Dokladna tylko dla instancji z brute-force."""
    refs: dict[str, tuple[float, bool]] = {}
    for instance in {r["instance"] for r in records}:
        exact = exact_optimum(instance)
        if exact is not None:
            refs[instance] = (exact[0], True)
        else:
            refs[instance] = (min(r["best_f"] for r in records if r["instance"] == instance), False)
    return refs


def group_summary(records: list[dict]) -> list[dict]:
    """Jeden wiersz na konfiguracje (algorytm, instancja, zestaw parametrow, N, I)."""
    refs = reference_values(records)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[tuple(r[k] for k in GROUP_KEYS)].append(r)

    rows = []
    for key, runs in groups.items():
        row = dict(zip(GROUP_KEYS, key))
        row.update(json.loads(runs[0]["params"]))
        f = [r["best_f"] for r in runs]
        row.update(describe_values(f))
        ref_f, exact = refs[row["instance"]]
        row["ref_f"] = ref_f
        row["ref_is_exact"] = exact
        for name in ("min", "median", "mean"):
            row[f"gap_{name}_pct"] = (row[f"f_{name}"] - ref_f) / ref_f * 100.0
        row["feasible_pct"] = 100.0 * float(np.mean([r["feasible"] for r in runs]))
        row["success_pct"] = (
            100.0 * float(np.mean([abs(x - ref_f) <= 1e-9 * abs(ref_f) for x in f])) if exact else ""
        )
        row["nfe_mean"] = float(np.mean([r["nfe"] for r in runs]))
        row["time_mean_s"] = float(np.mean([r["time_s"] for r in runs]))
        rows.append(row)
    rows.sort(key=lambda r: (r["algo"], r["instance"], r["set_id"], r["N"], r["I"]))
    return rows
