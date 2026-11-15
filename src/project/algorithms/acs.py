"""Ant Colony System (Dorigo & Gambardella, IEEE Trans. Evol. Comput. 1(1), 1997) dla wyboru k-podzbioru.

Adaptacja do problemu podzbioru: feromon tau_i lezy na *wezlach* (kolejnosc wyboru CH nie ma
znaczenia, wiec feromon na krawedziach rozmywalby informacje). Reszta to klasyczny ACS:

  * regula pseudolosowo-proporcjonalna: z prawdopodobienstwem q0 wybor argmax tau_i * eta_i^beta,
    w przeciwnym razie ruletka proporcjonalna do tau_i * eta_i^beta,
  * lokalna aktualizacja po kazdym wyborze: tau_i <- (1 - xi) * tau_i + xi * tau0,
  * globalna aktualizacja tylko dla najlepszego dotad rozwiazania: tau_i <- (1 - rho) * tau_i + rho / f_best,
  * tau0 = 1 / (k * f_ref), gdzie f_ref to rozwiazanie referencyjne (odpowiednik 1/(n * L_nn) z TSP).

Parametry bazowe z publikacji zrodlowej: beta = 2, rho = 0.1, xi = 0.1, q0 = 0.9.
N = liczba mrowek, I = liczba iteracji. NFE = N * I + 1 (jedna ewaluacja referencji dla tau0).
"""

from __future__ import annotations

import numpy as np

from project.core.algorithm import Algorithm
from project.core.evaluator import Evaluator
from project.core.problem import SubsetProblem


class AntColonySystem(Algorithm):
    name = "acs"

    def __init__(self, beta: float = 2.0, rho: float = 0.1, xi: float = 0.1, q0: float = 0.9) -> None:
        super().__init__(beta=beta, rho=rho, xi=xi, q0=q0)
        self.beta, self.rho, self.xi, self.q0 = beta, rho, xi, q0

    def _optimize(self, evaluator: Evaluator, n_agents: int, max_iter: int, rng: np.random.Generator) -> int:
        problem = evaluator.problem
        # Skala feromonu: liczona do NFE, ale nie jest kandydatem na wynik (track=False).
        f_ref = evaluator(problem.reference_solution(), track=False)
        tau0 = 1.0 / (problem.k * f_ref)
        tau = np.full(problem.n, tau0)
        for _ in range(max_iter):
            for _ant in range(n_agents):
                evaluator(self._construct(problem, tau, tau0, rng))
            best = evaluator.best_solution
            tau[best] = (1.0 - self.rho) * tau[best] + self.rho / evaluator.best_f
        return max_iter

    def _construct(self, problem: SubsetProblem, tau: np.ndarray, tau0: float, rng: np.random.Generator) -> np.ndarray:
        construction = problem.start_construction()
        chosen = np.zeros(problem.n, dtype=bool)
        solution = np.empty(problem.k, dtype=np.int64)
        for step in range(problem.k):
            score = tau * construction.eta() ** self.beta
            score[chosen] = 0.0
            if rng.random() < self.q0:
                node = int(score.argmax())
            else:
                cumulative = np.cumsum(score)
                node = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))
            chosen[node] = True
            solution[step] = node
            tau[node] = (1.0 - self.xi) * tau[node] + self.xi * tau0  # lokalna aktualizacja
            construction.add(node)
        return solution
