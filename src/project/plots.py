"""Wykresy do prezentacji i sprawozdania (Instrukcja, p. 5.3 i Krok 7).

    uv run python -m project.plots                      # wszystkie wykresy z katalogu results/
    uv run python -m project.plots --ref-n 40 --ref-i 40 --out results/figures

Wejscie: katalogi results/<algo>_<stage>_<instance> z runs.csv i histories.jsonl.
Kolory: ACS = niebieski, GC-BC = pomaranczowy (sloty 1-2 sprawdzonej palety), baseline losowy = szary.
Teksty maja kolor tekstu, nie kolor serii. Tozsamosc serii niesie legenda.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from project.tester.storage import load_runs  # noqa: E402

COLORS = {"acs": "#2a78d6", "gcbc": "#eb6834", "random": "#7a7974"}
LABELS = {"acs": "ACS", "gcbc": "GC-BC", "random": "Losowe (baseline)"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281", "#0d366b"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": GRID,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "lines.linewidth": 2.0,
    }
)


def _style(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  zapisano {path}")
    return path


# ------------------------------------------------------------------ wczytywanie
def load_all(results: Path, instance: str) -> tuple[list[dict], dict]:
    """Wszystkie uruchomienia (bez duplikatow) i metadane instancji z katalogow wynikow."""
    seen: dict[tuple, dict] = {}
    meta: dict = {}
    for algo in COLORS:
        for stage in ("baseline", "sensitivity", "all"):
            d = results / f"{algo}_{stage}_{instance}"
            if not (d / "runs.csv").exists():
                continue
            for r in load_runs(d, with_history=True):
                seen[(r["algo"], r["set_id"], r["N"], r["I"], r["run"])] = r
            if (d / "meta.json").exists() and not meta:
                meta = json.loads((d / "meta.json").read_text())["instance"]
    return list(seen.values()), meta


def _select(records: list[dict], algo: str, set_id: str = "base", n: int | None = None, i: int | None = None) -> list[dict]:
    return [
        r for r in records
        if r["algo"] == algo and r["set_id"] == set_id and (n is None or r["N"] == n) and (i is None or r["I"] == i)
    ]


# ------------------------------------------------------------------ Krok 7: zbieznosc f_best(NFE)
def plot_convergence(records: list[dict], meta: dict, n: int, i: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    top, bottom = [], []
    for algo in ("acs", "gcbc"):
        runs = _select(records, algo, "base", n, i)
        if not runs:
            continue
        # best-so-far to funkcja schodkowa po NFE; wartosci liczymy na wspolnej siatce NFE
        start = max(r["history"][0][0] for r in runs)
        end = max(r["nfe"] for r in runs)
        grid = np.arange(start, end + 1)
        curves = []
        for r in runs:
            nfe = np.array([h[0] for h in r["history"]])
            val = np.array([h[1] for h in r["history"]])
            curves.append(val[np.searchsorted(nfe, grid, side="right") - 1])
        curves = np.array(curves)
        q1, med, q3 = np.percentile(curves, [25, 50, 75], axis=0)
        ax.fill_between(grid, q1, q3, color=COLORS[algo], alpha=0.18, linewidth=0)
        ax.plot(grid, med, color=COLORS[algo], label=f"{LABELS[algo]} (mediana, pasmo = IQR, R={len(runs)})")
        top.append(med[0])
        bottom.append(q1.min())
    if "reference_f" in meta:
        ax.axhline(meta["reference_f"], color=INK2, linestyle="--", linewidth=1.2, label="rozwiązanie referencyjne (zachłanne)")
        bottom.append(meta["reference_f"])
    if top:  # start pojedynczych uruchomien bywa niedopuszczalny (f >> mediany), wiec przycinamy os do istotnego zakresu
        span = max(top) - min(bottom)
        ax.set_ylim(min(bottom) - 0.05 * span, max(top) + 0.05 * span)
    ax.set_xscale("log")
    ax.set_xlabel("liczba ewaluacji funkcji celu (NFE)")
    ax.set_ylabel("najlepsza dotąd wartość f")
    ax.set_title(f"Zbieżność f_best(NFE), N={n}, I={i}")
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out, f"convergence_N{n}_I{i}")


# ------------------------------------------------------------------ mapy ciepla N x I
def plot_heatmaps(records: list[dict], out: Path) -> None:
    algos = [a for a in ("acs", "gcbc") if _select(records, a)]
    if not algos:
        return
    n_values = sorted({r["N"] for r in records})
    i_values = sorted({r["I"] for r in records})
    mats = {}
    for algo in algos:
        m = np.full((len(n_values), len(i_values)), np.nan)
        for a, n in enumerate(n_values):
            for b, i in enumerate(i_values):
                runs = _select(records, algo, "base", n, i)
                if runs:
                    m[a, b] = np.median([r["best_f"] for r in runs])
        mats[algo] = m
    vmin, vmax = np.nanmin([m for m in mats.values()]), np.nanmax([m for m in mats.values()])
    fig, axes = plt.subplots(1, len(algos), figsize=(5.2 * len(algos), 3.8), squeeze=False)
    for ax, algo in zip(axes[0], algos):
        m = mats[algo]
        im = ax.imshow(m, cmap=BLUE_RAMP, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        ax.set_xticks(range(len(i_values)), i_values)
        ax.set_yticks(range(len(n_values)), n_values)
        ax.set_xlabel("liczba iteracji I")
        ax.set_ylabel("liczebność populacji N")
        ax.set_title(LABELS[algo])
        for a in range(m.shape[0]):
            for b in range(m.shape[1]):
                if not math.isnan(m[a, b]):
                    dark = (m[a, b] - vmin) / (vmax - vmin + 1e-30) > 0.5
                    ax.text(b, a, f"{m[a, b]:.4f}", ha="center", va="center", fontsize=8, color="#ffffff" if dark else INK)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.colorbar(im, ax=axes[0].tolist(), label="mediana f (ciemniej = gorzej)", shrink=0.85)
    fig.suptitle("Mediana najlepszego f po R uruchomieniach, zestaw bazowy", x=0.01, ha="left", fontweight="bold")
    _save(fig, out, "heatmap_N_I")


# ------------------------------------------------------------------ rozklad wynikow
def plot_boxplot(records: list[dict], n: int, i: int, out: Path) -> None:
    algos = [a for a in ("acs", "gcbc", "random") if _select(records, a, "base", n, i)]
    if not algos:
        return
    # Lewy panel: wszystkie serie w skali log (baseline losowy jest o rzad wielkosci gorszy).
    # Prawy panel: zblizenie na metaheurystyki w skali liniowej, gdzie widac roznice miedzy nimi.
    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    rng = np.random.default_rng(0)

    def draw(ax, names: list[str]) -> None:
        for pos, algo in enumerate(names):
            f = np.array([r["best_f"] for r in _select(records, algo, "base", n, i)])
            ax.boxplot(
                [f], positions=[pos], widths=0.5, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=COLORS[algo], alpha=0.25, edgecolor=COLORS[algo], linewidth=1.5),
                medianprops=dict(color=INK, linewidth=2),
                whiskerprops=dict(color=COLORS[algo], linewidth=1.5), capprops=dict(color=COLORS[algo], linewidth=1.5),
            )
            ax.scatter(pos + rng.uniform(-0.12, 0.12, len(f)), f, s=28, color=COLORS[algo], edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.set_xticks(range(len(names)), [LABELS[a] for a in names])
        _style(ax)

    draw(ax_all, algos)
    ax_all.set_yscale("log")
    ax_all.set_ylabel("końcowe f (skala logarytmiczna)")
    ax_all.set_title(f"Wszystkie metody, N={n}, I={i}")
    zoom = [a for a in algos if a != "random"]
    draw(ax_zoom, zoom)
    ax_zoom.set_ylabel("końcowe f (skala liniowa)")
    ax_zoom.set_title("Zbliżenie: metaheurystyki")
    _save(fig, out, f"boxplot_N{n}_I{i}")


# ------------------------------------------------------------------ jakosc vs koszt
def plot_quality_vs_cost(records: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for algo in ("acs", "gcbc"):
        pts = {}
        for r in _select(records, algo):
            pts.setdefault((r["N"], r["I"]), []).append(r)
        if not pts:
            continue
        x = np.array([np.mean([r["nfe"] for r in v]) for v in pts.values()])
        y = np.array([np.median([r["best_f"] for r in v]) for v in pts.values()])
        ax.scatter(x, y, s=34, color=COLORS[algo], edgecolor=SURFACE, linewidth=1, zorder=3, label=f"{LABELS[algo]} (każda para N, I)")
        order = np.argsort(x)
        envelope = np.minimum.accumulate(y[order])  # najlepsza mediana osiagalna do danego NFE
        ax.step(x[order], envelope, where="post", color=COLORS[algo], linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("średnia liczba ewaluacji (NFE)")
    ax.plot([], [], color=INK2, linewidth=1.6, label="obwiednia: najlepsza mediana do danego NFE")
    ax.set_ylabel("mediana f")
    ax.set_title("Jakość a koszt obliczeniowy")
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out, "quality_vs_cost")


# ------------------------------------------------------------------ wrazliwosc na parametry
def _sort_key(v):
    return (1, 0.0, "") if isinstance(v, float) and math.isinf(v) else (0, v, "") if isinstance(v, (int, float)) else (2, 0.0, str(v))


def plot_sensitivity(records: list[dict], algo: str, n: int, i: int, out: Path) -> None:
    at_budget = [r for r in records if r["algo"] == algo and r["N"] == n and r["I"] == i]
    params = {r["set_id"]: json.loads(r["params"]) for r in at_budget}
    if "base" not in params or len(params) < 2:
        return
    base = params["base"]
    panels: dict[str, list[tuple]] = {}
    for set_id, p in params.items():
        diff = [k for k in p if p[k] != base[k]]
        if set_id == "base":
            for k in base:
                panels.setdefault(k, []).append((base[k], set_id))
        elif len(diff) == 1:
            panels.setdefault(diff[0], []).append((p[diff[0]], set_id))
    panels = {k: v for k, v in panels.items() if len(v) > 1}

    stats = {}
    for set_id in params:
        f = np.array([r["best_f"] for r in at_budget if r["set_id"] == set_id])
        stats[set_id] = np.percentile(f, [25, 50, 75])
    # Wartosci o rzad wielkosci gorsze od bazowej (np. ablacja init=random) splaszczylyby wspolna os Y,
    # wiec nie rysujemy ich w panelach i opisujemy w przypisie (pelne dane sa w summary.csv).
    base_median = stats["base"][1]
    off_scale = {sid: stats[sid][1] for sid in stats if stats[sid][1] > 3 * base_median}
    footnote = "; ".join(f"{sid}: mediana {med:.3g}" for sid, med in sorted(off_scale.items()))
    panels = {
        name: [it for it in items if it[1] not in off_scale] for name, items in panels.items()
    }
    panels = {k: v for k, v in panels.items() if len(v) > 1}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.6), sharey=True, squeeze=False)
    for ax, (name, items) in zip(axes[0], sorted(panels.items())):
        items = sorted(items, key=lambda t: _sort_key(t[0]))
        xs = np.arange(len(items))
        q1, med, q3 = (np.array([stats[sid][j] for _, sid in items]) for j in range(3))
        ax.errorbar(xs, med, yerr=[med - q1, q3 - med], color=COLORS[algo], marker="o", markersize=8, capsize=3, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        labels = ["inf" if isinstance(v, float) and math.isinf(v) else str(v) for v, _ in items]
        labels = [lab + "\n(baza)" if sid == "base" else lab for lab, (_, sid) in zip(labels, items)]
        ax.set_xticks(xs, labels, fontsize=8)
        ax.set_xlim(-0.5, len(items) - 0.5)
        ax.set_title(name)
        _style(ax)
    axes[0][0].set_ylabel("mediana f (słupki = IQR)")
    fig.suptitle(f"Wrażliwość {LABELS[algo]} na parametry wewnętrzne, N={n}, I={i}", x=0.01, ha="left", fontweight="bold")
    if footnote:
        fig.text(0.01, -0.04, f"Poza skalą wykresu (zestawy znacznie gorsze od bazowego): {footnote}", fontsize=8.5, color=INK2)
    _save(fig, out, f"sensitivity_{algo}_N{n}_I{i}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/figures")
    p.add_argument("--instance", default="main")
    p.add_argument("--ref-n", type=int, default=40)
    p.add_argument("--ref-i", type=int, default=40)
    args = p.parse_args(argv)
    records, meta = load_all(Path(args.results), args.instance)
    if not records:
        raise SystemExit(f"brak wynikow w {args.results} dla instancji {args.instance}")
    out = Path(args.out)
    print(f"{len(records)} uruchomien")
    plot_convergence(records, meta, args.ref_n, args.ref_i, out)
    plot_heatmaps(records, out)
    plot_boxplot(records, args.ref_n, args.ref_i, out)
    plot_quality_vs_cost(records, out)
    for algo in ("acs", "gcbc"):
        plot_sensitivity(records, algo, args.ref_n, args.ref_i, out)


if __name__ == "__main__":
    main()
