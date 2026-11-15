"""Losowe przeszukiwanie. Tylko baseline do sprawdzenia, czy metaheurystyki cokolwiek daja."""

from __future__ import annotations

import numpy as np

from project.core.algorithm import Algorithm
from project.core.evaluator import Evaluator


class RandomSearch(Algorithm):
    name = "random"

    def _optimize(self, evaluator: Evaluator, n_agents: int, max_iter: int, rng: np.random.Generator) -> int:
        problem = evaluator.problem
        for _ in range(max_iter):
            for _ in range(n_agents):
                evaluator(problem.random_solution(rng))
        return max_iter
