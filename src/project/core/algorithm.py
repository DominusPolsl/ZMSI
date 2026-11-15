"""Wspolny interfejs algorytmow. Tester zna tylko `Algorithm.run` i `RunResult`."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from project.core.evaluator import BudgetExhausted, Evaluator
from project.core.problem import SubsetProblem


@dataclass
class RunResult:
    best_f: float
    best_solution: list[int]
    nfe: int
    iterations: int
    seed: int
    time_s: float
    # (nfe, f_best) przy kazdej poprawie best-so-far
    history: list[tuple[int, float]] = field(default_factory=list)


class Algorithm(ABC):
    """Metaheurystyka dla `SubsetProblem`.

    Parametry wewnetrzne (P1..Pk) podaje sie w konstruktorze, a parametry wspolne
    (N, I, seed, opcjonalnie budzet NFE) w `run`.
    """

    name: str = "algorithm"

    def __init__(self, **params) -> None:
        self.params = params

    def run(
        self,
        problem: SubsetProblem,
        n_agents: int,
        max_iter: int,
        seed: int,
        max_nfe: int | None = None,
    ) -> RunResult:
        rng = np.random.default_rng(seed)
        evaluator = Evaluator(problem, max_nfe=max_nfe)
        iterations = 0
        t0 = time.perf_counter()
        try:
            iterations = self._optimize(evaluator, n_agents, max_iter, rng)
        except BudgetExhausted:
            pass  # kryterium stopu: wyczerpany budzet NFE
        elapsed = time.perf_counter() - t0
        best = evaluator.best_solution  # None tylko gdy budzet NFE skonczyl sie przed pierwszym rozwiazaniem
        return RunResult(
            best_f=evaluator.best_f,
            best_solution=[] if best is None else [int(i) for i in best],
            nfe=evaluator.nfe,
            iterations=iterations,
            seed=seed,
            time_s=elapsed,
            history=list(evaluator.history),
        )

    @abstractmethod
    def _optimize(
        self,
        evaluator: Evaluator,
        n_agents: int,
        max_iter: int,
        rng: np.random.Generator,
    ) -> int:
        """Wykonuje optymalizacje (ewaluacje wylacznie przez `evaluator`).

        Zwraca liczbe ukonczonych iteracji. Kryterium stopu: max_iter (lub max_nfe w `run`).
        """
