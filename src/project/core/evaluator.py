"""Ewaluator: jedyne miejsce, w ktorym liczone sa NFE i sledzone best-so-far.

Algorytm wola `evaluator(solution)` zamiast `problem.evaluate(...)`. Dzieki temu NFE jest
liczone dokladnie i identycznie dla kazdego algorytmu (Instrukcja testowania, Krok 2 i 7).
"""

from __future__ import annotations

import numpy as np

from project.core.problem import SubsetProblem


class BudgetExhausted(Exception):
    """Zglaszany, gdy algorytm probuje wykonac ewaluacje ponad limit max_nfe."""


class Evaluator:
    def __init__(self, problem: SubsetProblem, max_nfe: int | None = None) -> None:
        self.problem = problem
        self.max_nfe = max_nfe
        self.nfe = 0
        self.best_f = float("inf")
        self.best_solution: np.ndarray | None = None
        # Zdarzenia poprawy best-so-far: (nfe, f_best). Pelna krzywa f_best(NFE) jest funkcja schodkowa.
        self.history: list[tuple[int, float]] = []

    def __call__(self, solution: np.ndarray, track: bool = True) -> float:
        """Ewaluacja liczona do NFE. `track=False` to ewaluacja pomocnicza (np. skala feromonu):
        zwiekszamy NFE, ale rozwiazanie nie trafia do best-so-far, wiec nie jest wynikiem algorytmu."""
        if self.max_nfe is not None and self.nfe >= self.max_nfe:
            raise BudgetExhausted
        f = float(self.problem.evaluate(solution))
        self.nfe += 1
        if track and f < self.best_f:
            self.best_f = f
            self.best_solution = np.sort(np.asarray(solution, dtype=np.int64))
            self.history.append((self.nfe, f))
        return f
