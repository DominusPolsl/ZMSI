"""Wykonywanie uruchomien: zestaw_parametrow x N x I x R, z zapisanym seedem kazdego uruchomienia."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from project.problems.wsn_ch import INSTANCES, WSNClusterHeadProblem, brute_force
from project.tester.config import BASE_SEED, ParameterSet


@dataclass(frozen=True)
class RunTask:
    algo: str
    set_id: str
    params: tuple
    instance: str
    n_agents: int
    max_iter: int
    run: int
    seed: int
    max_nfe: int | None = None
    opt_solution: tuple | None = None  # znane optimum (tylko instancja "tiny")


@lru_cache(maxsize=None)
def get_problem(name: str) -> WSNClusterHeadProblem:
    return INSTANCES[name]()


@lru_cache(maxsize=None)
def exact_optimum(instance: str) -> tuple[float, tuple[int, ...]] | None:
    """Prawdziwe optimum z brute-force, tylko dla malej instancji."""
    if instance != "tiny":
        return None
    f, sol = brute_force(get_problem(instance))
    return f, tuple(int(i) for i in sorted(sol))


def make_tasks(
    sets: list[ParameterSet],
    instance: str,
    n_values: tuple[int, ...],
    i_values: tuple[int, ...],
    runs: int,
    base_seed: int = BASE_SEED,
    max_nfe: int | None = None,
) -> list[RunTask]:
    opt = exact_optimum(instance)
    opt_solution = opt[1] if opt else None
    return [
        RunTask(ps.algo, ps.set_id, ps.params, instance, n, i, r, base_seed + r, max_nfe, opt_solution)
        for ps in sets
        for n in n_values
        for i in i_values
        for r in range(runs)
    ]


def execute_run(task: RunTask) -> dict:
    """Jedno uruchomienie. Funkcja top-level, zeby dzialala z ProcessPoolExecutor."""
    from project.tester.config import ALGORITHMS

    problem = get_problem(task.instance)
    algorithm = ALGORITHMS[task.algo](**dict(task.params))
    result = algorithm.run(problem, task.n_agents, task.max_iter, task.seed, max_nfe=task.max_nfe)
    solution = np.array(result.best_solution, dtype=np.int64)
    mismatch = ""
    if task.opt_solution is not None:
        mismatch = problem.k - len(set(result.best_solution) & set(task.opt_solution))
    return {
        "algo": task.algo,
        "set_id": task.set_id,
        "params": json.dumps(dict(task.params)),
        "instance": task.instance,
        "N": task.n_agents,
        "I": task.max_iter,
        "run": task.run,
        "seed": task.seed,
        "max_nfe": "" if task.max_nfe is None else task.max_nfe,
        "nfe": result.nfe,
        "iterations": result.iterations,
        "best_f": result.best_f,
        "feasible": bool(len(solution) == problem.k and problem.is_feasible(solution)),
        "set_mismatch": mismatch,
        "time_s": result.time_s,
        "best_solution": " ".join(map(str, result.best_solution)),
        "history": result.history,
    }


def run_tasks(tasks: list[RunTask], workers: int = 1, label: str = "") -> list[dict]:
    total = len(tasks)
    records: list[dict] = []
    t0 = time.perf_counter()
    step = max(1, total // 20)

    def progress(done: int) -> None:
        if done % step == 0 or done == total:
            print(f"  {label} {done}/{total} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr, flush=True)

    if workers <= 1:
        for idx, task in enumerate(tasks, 1):
            records.append(execute_run(task))
            progress(idx)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for idx, record in enumerate(pool.map(execute_run, tasks, chunksize=4), 1):
                records.append(record)
                progress(idx)
    return records
