"""Testy istotnosci roznic miedzy konfiguracjami (Instrukcja: wnioski z serii uruchomien, nie z pojedynczego wyniku).

    uv run python -m project.significance                              # wyniki z katalogu results/
    uv run python -m project.significance --ref-n 40 --ref-i 40 --out results/significance

Wejscie: katalogi results/<algo>_<stage>_<instance> z runs.csv. Wyjscie: trzy pliki CSV w --out i tabele na stdout.

Test: dwustronny, *dokladny* test permutacyjny sumy rang (to samo co test Manna-Whitneya U, z rangami srednimi przy
remisach). Rozklad sumy rang pod H0 liczony programowaniem dynamicznym po sumach rang (podwojone rangi sa calkowite,
wiec remisy nie psuja dokladnosci), bez scipy i bez przyblizenia normalnego. Probki sa niezalezne (seedy 1000..1009
maja te same wartosci we wszystkich konfiguracjach, ale algorytmy zuzywaja je inaczej, wiec nie parujemy).
Miara efektu: A12 Vargha-Delaneya = P(X < Y) + 0,5 P(X = Y); f minimalizujemy, wiec A12 > 0,5 znaczy, ze X jest lepsze.

Zestawienia:
  (a) OAT: kazdy wariant parametru vs zestaw bazowy (ten sam algorytm, N = ref-n, I = ref-i), korekta Holma w obrebie algorytmu,
  (b) ACS vs GC-BC (zestawy bazowe) dla N = I = ref oraz 80,
  (c) ACS vs GC-BC przy zblizonym NFE: dla kazdej komorki (N, I) ACS bierzemy komorke GC-BC o najnizszej medianie sposrod
      tych, ktorych srednie NFE rozni sie o mniej niz --nfe-tol (korzystne dla GC-BC, wiec wniosek jest ostrozny), Holm w calej tabeli.
"""

from __future__ import annotations

import argparse
import statistics
from math import comb
from pathlib import Path

from project.tester.storage import load_runs, write_table

ALGOS = ("acs", "gcbc", "random")
Group = dict  # {"f": list[float], "nfe": float}


# ------------------------------------------------------------------ testy
def _doubled_midranks(values: list[float]) -> list[int]:
    """Podwojone rangi srednie (calkowite): rowne wartosci dostaja srednia z zajmowanych pozycji."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks2 = [0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks2[order[t]] = i + j + 2  # srednia rang (i+1 .. j+1) razy 2
        i = j + 1
    return ranks2


def rank_sum_pvalue(x: list[float], y: list[float]) -> float:
    """Dwustronna wartosc p dokladnego testu permutacyjnego sumy rang (rownowazna testowi Manna-Whitneya U)."""
    n, m = len(x), len(y)
    total_n = n + m
    ranks2 = _doubled_midranks(list(x) + list(y))
    s2 = sum(ranks2)
    observed = sum(ranks2[:n])
    # counts[j][s] = liczba j-elementowych podzbiorow o sumie podwojonych rang s (plecak 0/1, j malejaco)
    counts = [[0] * (s2 + 1) for _ in range(n + 1)]
    counts[0][0] = 1
    for r in ranks2:
        for j in range(n, 0, -1):
            prev, cur = counts[j - 1], counts[j]
            for s in range(s2, r - 1, -1):
                if prev[s - r]:
                    cur[s] += prev[s - r]
    deviation = abs(observed * total_n - n * s2)  # |T - E T| przeskalowane do calkowitych
    hits = sum(c for s, c in enumerate(counts[n]) if abs(s * total_n - n * s2) >= deviation)
    return hits / comb(total_n, n)


def a12(x: list[float], y: list[float]) -> float:
    """Vargha-Delaney: P(X < Y) + 0,5 P(X = Y). Przy minimalizacji wartosc > 0,5 oznacza, ze X jest lepsze."""
    score = sum((a < b) + 0.5 * (a == b) for a in x for b in y)
    return score / (len(x) * len(y))


def holm(pvalues: list[float]) -> list[float]:
    """Skorygowane wartosci p metoda Holma (step-down), w kolejnosci wejscia."""
    m = len(pvalues)
    order = sorted(range(m), key=pvalues.__getitem__)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


# ------------------------------------------------------------------ dane
def load_groups(results: Path, instance: str) -> dict[tuple, Group]:
    """(algo, set_id, N, I) -> {"f": najlepsze f z uruchomien (wg numeru run), "nfe": srednie NFE}."""
    seen: dict[tuple, dict] = {}
    for algo in ALGOS:
        for stage in ("baseline", "sensitivity", "all"):
            d = results / f"{algo}_{stage}_{instance}"
            if (d / "runs.csv").exists():
                for r in load_runs(d):
                    seen[(r["algo"], r["set_id"], r["N"], r["I"], r["run"])] = r
    groups: dict[tuple, Group] = {}
    for key in sorted(seen):
        r = seen[key]
        g = groups.setdefault(key[:4], {"f": [], "nfe": []})
        g["f"].append(r["best_f"])
        g["nfe"].append(r["nfe"])
    for g in groups.values():
        g["nfe"] = sum(g["nfe"]) / len(g["nfe"])
    return groups


def _compare(x: Group, y: Group) -> dict:
    return {
        "R_x": len(x["f"]),
        "R_y": len(y["f"]),
        "median_x": statistics.median(x["f"]),
        "median_y": statistics.median(y["f"]),
        "p": rank_sum_pvalue(x["f"], y["f"]),
        "a12": a12(x["f"], y["f"]),
    }


# ------------------------------------------------------------------ zestawienia
def oat_vs_base(groups: dict[tuple, Group], n: int, i: int, alpha: float) -> list[dict]:
    rows: list[dict] = []
    for algo in ("acs", "gcbc"):
        base = groups.get((algo, "base", n, i))
        if base is None:
            continue
        sets = sorted(s for (a, s, nn, ii) in groups if a == algo and nn == n and ii == i and s != "base")
        part = []
        for set_id in sets:
            c = _compare(groups[(algo, set_id, n, i)], base)  # x = wariant, y = baza
            part.append(
                {
                    "algo": algo, "set_id": set_id, "N": n, "I": i,
                    "median_variant": c["median_x"], "median_base": c["median_y"],
                    "delta_median_pct": (c["median_x"] / c["median_y"] - 1.0) * 100.0,
                    "p": c["p"], "a12_variant_better": c["a12"],
                }
            )
        for row, p_adj in zip(part, holm([r["p"] for r in part])):
            row["p_holm"] = p_adj
            row["significant"] = p_adj < alpha
        rows.extend(part)
    return rows


def acs_vs_gcbc(groups: dict[tuple, Group], cells: list[tuple[int, int]]) -> list[dict]:
    rows = []
    for n, i in cells:
        a, g = groups.get(("acs", "base", n, i)), groups.get(("gcbc", "base", n, i))
        if a is None or g is None:
            continue
        c = _compare(a, g)
        rows.append(
            {
                "N": n, "I": i, "nfe_acs": a["nfe"], "nfe_gcbc": g["nfe"],
                "median_acs": c["median_x"], "median_gcbc": c["median_y"],
                "p": c["p"], "a12_acs_better": c["a12"],
            }
        )
    return rows


def equal_nfe(groups: dict[tuple, Group], tol: float, alpha: float) -> list[dict]:
    gc = {(n, i): g for (algo, s, n, i), g in groups.items() if algo == "gcbc" and s == "base"}
    rows = []
    for (algo, s, n, i), a in sorted(groups.items()):
        if algo != "acs" or s != "base":
            continue
        near = [(k, g) for k, g in gc.items() if abs(g["nfe"] - a["nfe"]) / a["nfe"] < tol]
        if not near:
            continue
        (gn, gi), g = min(near, key=lambda kg: statistics.median(kg[1]["f"]))
        c = _compare(a, g)
        rows.append(
            {
                "N_acs": n, "I_acs": i, "nfe_acs": a["nfe"], "N_gcbc": gn, "I_gcbc": gi, "nfe_gcbc": g["nfe"],
                "median_acs": c["median_x"], "median_gcbc": c["median_y"],
                "p": c["p"], "a12_acs_better": c["a12"],
            }
        )
    for row, p_adj in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = p_adj
        row["significant"] = p_adj < alpha
    return rows


def _show(title: str, rows: list[dict], columns: list[str]) -> None:
    print(f"\n== {title}")
    print("  ".join(f"{c:>16s}" for c in columns))
    for r in rows:
        cells = []
        for c in columns:
            v = r[c]
            cells.append(f"{v:>16.5g}" if isinstance(v, float) else f"{str(v):>16s}")
        print("  ".join(cells))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/significance")
    p.add_argument("--instance", default="main")
    p.add_argument("--ref-n", type=int, default=40)
    p.add_argument("--ref-i", type=int, default=40)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--nfe-tol", type=float, default=0.06, help="okno wzglednej roznicy NFE w zestawieniu (c)")
    args = p.parse_args(argv)

    groups = load_groups(Path(args.results), args.instance)
    if not groups:
        raise SystemExit(f"brak wynikow w {args.results} dla instancji {args.instance}")
    out = Path(args.out)

    oat = oat_vs_base(groups, args.ref_n, args.ref_i, args.alpha)
    cells = sorted({(args.ref_n, args.ref_i), (80, 80)})
    pair = acs_vs_gcbc(groups, cells)
    near = equal_nfe(groups, args.nfe_tol, args.alpha)

    write_table(out / "significance_oat.csv", oat)
    write_table(out / "significance_acs_vs_gcbc.csv", pair)
    write_table(out / "significance_equal_nfe.csv", near)

    _show(
        f"(a) OAT vs baza, N={args.ref_n}, I={args.ref_i} (Holm w obrebie algorytmu, alfa={args.alpha})", oat,
        ["algo", "set_id", "median_variant", "median_base", "delta_median_pct", "p", "p_holm", "a12_variant_better", "significant"],
    )
    _show(
        "(b) ACS vs GC-BC, zestawy bazowe, ten sam (N, I)", pair,
        ["N", "I", "nfe_acs", "nfe_gcbc", "median_acs", "median_gcbc", "p", "a12_acs_better"],
    )
    _show(
        f"(c) ACS vs najlepsza komorka GC-BC o zblizonym NFE (okno {args.nfe_tol:.0%}, Holm w tabeli)", near,
        ["N_acs", "I_acs", "nfe_acs", "N_gcbc", "I_gcbc", "nfe_gcbc", "median_acs", "median_gcbc", "p", "p_holm", "a12_acs_better"],
    )
    print(f"\nzapisano CSV w {out}/")


if __name__ == "__main__":
    main()
