"""CLI testera.

Przyklady (z katalogu glownego projektu):
    uv run python -m project.tester.run_experiments --stage verify
    uv run python -m project.tester.run_experiments --stage baseline --algo acs gcbc --workers 4
    uv run python -m project.tester.run_experiments --stage sensitivity --algo acs --workers 4
    uv run python -m project.tester.run_experiments --stage summarize --inputs results/acs_all_main results/gcbc_all_main
Szybki przebieg kontrolny: dodaj --runs 3 --n-values 10,20 --i-values 5,10.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from project.problems.wsn_ch import INSTANCES
from project.tester import config
from project.tester.runner import get_problem, make_tasks, run_tasks
from project.tester.selection import best_per_budget
from project.tester.stats import group_summary
from project.tester.storage import load_runs, save_runs, write_meta, write_table


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def _print_best(rows: list[dict], ref_n: int, ref_i: int) -> None:
    winners = best_per_budget(rows)
    print(f"\nNajlepszy zestaw parametrow przy N={ref_n}, I={ref_i} (kryterium: min mediana, potem s, potem NFE):")
    for w in winners:
        if w["N"] == ref_n and w["I"] == ref_i:
            print(
                f"  {w['algo']:6s} {w['instance']:5s} zestaw={w['set_id']:12s} mediana={w['f_median']:.6g} "
                f"s={w['f_std']:.3g} CV={w['f_cv_pct']:.2f}% gap_med={w['gap_median_pct']:.3f}% NFE={w['nfe_mean']:.0f}"
            )
    print("\nNajlepsza konfiguracja ogolnie (min mediana po wszystkich zestawach i budzetach):")
    for algo in sorted({r["algo"] for r in rows}):
        best = min((r for r in rows if r["algo"] == algo), key=lambda r: (r["f_median"], r["f_std"], r["nfe_mean"]))
        print(
            f"  {algo:6s} zestaw={best['set_id']} N={best['N']} I={best['I']} NFE={best['nfe_mean']:.0f} "
            f"mediana={best['f_median']:.6g} s={best['f_std']:.3g}"
        )


def summarize(dirs: list[Path], out: Path, ref_n: int, ref_i: int) -> None:
    records = [r for d in dirs for r in load_runs(d)]
    rows = group_summary(records)
    write_table(out / "summary.csv", rows)
    write_table(out / "best_configs.csv", best_per_budget(rows))
    print(f"Zapisano {len(rows)} konfiguracji w {out / 'summary.csv'}")
    _print_best(rows, ref_n, ref_i)


def run_stage(algo: str, args: argparse.Namespace) -> Path:
    sets = config.select_sets(algo, args.stage)
    out = Path(args.out_root) / f"{algo}_{args.stage}_{args.instance}"
    tasks = make_tasks(sets, args.instance, args.n_values, args.i_values, args.runs, args.base_seed, args.max_nfe)
    print(f"[{algo}] {len(sets)} zestawow x {len(args.n_values)} N x {len(args.i_values)} I x {args.runs} R = {len(tasks)} uruchomien")
    records = run_tasks(tasks, args.workers, label=algo)
    save_runs(out, records)
    write_meta(
        out,
        stage=args.stage,
        algo=algo,
        instance=get_problem(args.instance).describe(),
        parameter_sets=[{"set_id": s.set_id, "params": s.as_dict} for s in sets],
        n_values=args.n_values,
        i_values=args.i_values,
        runs=args.runs,
        base_seed=args.base_seed,
        seeds=[args.base_seed + r for r in range(args.runs)],
        max_nfe=args.max_nfe,
        stop_criterion="max_iter" + (" lub max_nfe" if args.max_nfe else ""),
    )
    write_table(out / "summary.csv", group_summary(records))
    print(f"[{algo}] zapisano wyniki w {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["verify", "baseline", "sensitivity", "all", "summarize"])
    p.add_argument("--algo", nargs="+", default=["acs", "gcbc"], choices=sorted(config.ALGORITHMS))
    p.add_argument("--instance", default="main", choices=sorted(INSTANCES))
    p.add_argument("--runs", type=int, default=config.R)
    p.add_argument("--n-values", type=_ints, default=config.N_VALUES)
    p.add_argument("--i-values", type=_ints, default=config.I_VALUES)
    p.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    p.add_argument("--max-nfe", type=int, default=None, help="opcjonalny wspolny limit NFE (porownania przy rownym budzecie)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out-root", default="results")
    p.add_argument("--inputs", nargs="+", help="katalogi z runs.csv dla --stage summarize")
    p.add_argument("--out", default="results/summary", help="katalog wyjsciowy dla --stage summarize")
    p.add_argument("--ref-n", type=int, default=40)
    p.add_argument("--ref-i", type=int, default=40)
    args = p.parse_args(argv)

    if args.stage == "verify":
        from project.tester.sanity import run_sanity

        return 0 if run_sanity() else 1
    if args.stage == "summarize":
        if not args.inputs:
            p.error("--stage summarize wymaga --inputs")
        summarize([Path(d) for d in args.inputs], Path(args.out), args.ref_n, args.ref_i)
        return 0

    outs = [run_stage(algo, args) for algo in args.algo]
    summarize(outs, Path(args.out_root) / f"summary_{args.stage}_{args.instance}", args.ref_n, args.ref_i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
