"""Wybor najlepszej konfiguracji wg jawnego kryterium (Instrukcja, Krok 6).

Kryterium (przy stalych N i I, czyli stalym budzecie):
  1. najmniejsza mediana f_r,
  2. przy medianach zblizonych (w granicach TIE_TOL wzglednie od najlepszej) mniejsze rozproszenie s_f,
  3. przy rownym rozproszeniu mniejszy koszt NFE.
Konfiguracji nie wybieramy na podstawie pojedynczego najlepszego trafienia (f_min).
"""

from __future__ import annotations

TIE_TOL = 0.005  # 0.5 %


def pick_best(rows: list[dict]) -> dict:
    best_median = min(r["f_median"] for r in rows)
    close = [r for r in rows if r["f_median"] <= best_median * (1.0 + TIE_TOL)]
    return min(close, key=lambda r: (r["f_std"], r["nfe_mean"]))


def best_per_budget(rows: list[dict]) -> list[dict]:
    """Zwyciezca (zestaw parametrow) dla kazdej trojki (algorytm, instancja, N, I)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["algo"], r["instance"], r["N"], r["I"]), []).append(r)
    return [pick_best(g) for _, g in sorted(groups.items())]
