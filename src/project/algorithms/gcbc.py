"""Graph Constrained Bee Colony (GC-BC): dyskretny ABC z sasiedztwem ograniczonym grafem.

UWAGA: algorytm o tej nazwie nie wystepuje w literaturze. To *autorski wariant* dyskretnego ABC
(nazwa robocza), zbudowany na:
  * Karaboga, Basturk, J. Global Optim. 39, 2007 (fazy employed / onlooker / scout, limit prob),
  * ABC dla problemow grafowych, np. Singh, Appl. Soft Comput. 9(2), 2009 (leaf-constrained MST).

Rozwiazanie (zrodlo pokarmu) to k-podzbior wezlow. Odpowiednik ruchu v = x + phi * (x - x_m) z
ciaglego ABC jest dyskretny i *ograniczony grafem G* (krawedz <=> odleglosc <= radius * d0):
wybierz CH u z rozwiazania i partnera m; zastap u wezlem v nalezacym do rozwiazania partnera
(poza biezacym), ktory jest sasiadem u w G. Gdy takiego v brak, zastap u dowolnym sasiadem z G
spoza rozwiazania. CH bez zadnego dostepnego sasiada zostaje na miejscu.

Parametry wewnetrzne:
  limit    - liczba nieudanych prob, po ktorej zrodlo jest porzucane (scout),
  n_swaps  - liczba zamian CH w jednym ruchu,
  radius   - promien grafu w jednostkach d0 (np.inf = graf pelny, czyli brak ograniczenia),
  init     - start i scouty: "heuristic" (losowa konstrukcja zachlanna ~ eta^2, ta sama heurystyka
             co w ACS) lub "random" (losowy k-podzbior). Przy starcie losowym populacja nie osiaga
             rozwiazan dopuszczalnych nawet w najwiekszym budzecie siatki (N=80, I=80, ok. 12.9 tys. NFE;
             sprawdzone na instancji "main"), wiec domyslnie "heuristic". "random" zostaje jako ablacja.

N = liczba zrodel pokarmu (= pszczol zatrudnionych = obserwatorek), I = liczba iteracji.
NFE = N (inicjalizacja) + I * (2N + liczba scoutow); w kazdej iteracji najwyzej jeden scout.
Selekcja obserwatorek jest niezalezna od skali f (znormalizowana ruletka), bo klasyczne 1/(1+f)
jest prawie jednostajne dla f rzedu 0.1.
"""

from __future__ import annotations

import numpy as np

from project.core.algorithm import Algorithm
from project.core.evaluator import Evaluator
from project.core.problem import SubsetProblem


class GraphConstrainedBeeColony(Algorithm):
    name = "gcbc"

    def __init__(self, limit: int = 20, n_swaps: int = 1, radius: float = 1.0, init: str = "heuristic") -> None:
        super().__init__(limit=limit, n_swaps=n_swaps, radius=radius, init=init)
        assert init in ("heuristic", "random")
        self.limit, self.n_swaps, self.radius, self.init = limit, n_swaps, radius, init

    def _new_source(self, problem: SubsetProblem, rng: np.random.Generator) -> np.ndarray:
        if self.init == "random":
            return problem.random_solution(rng)
        return problem.heuristic_solution(rng)

    def _optimize(self, evaluator: Evaluator, n_agents: int, max_iter: int, rng: np.random.Generator) -> int:
        problem = evaluator.problem
        n_src = n_agents
        adj = problem.adjacency(self.radius)

        foods = np.array([self._new_source(problem, rng) for _ in range(n_src)], dtype=np.int64)
        member = np.zeros((n_src, problem.n), dtype=bool)
        np.put_along_axis(member, foods, True, axis=1)
        fit = np.array([evaluator(foods[i]) for i in range(n_src)])
        trials = np.zeros(n_src, dtype=np.int64)

        def try_improve(i: int) -> None:
            partner = int(rng.integers(n_src - 1))
            partner += partner >= i  # partner != i
            cand, cand_member = self._neighbor(foods[i], member[i], member[partner], adj, rng)
            f_new = evaluator(cand)
            if f_new < fit[i]:  # selekcja zachlanna
                foods[i], member[i], fit[i], trials[i] = cand, cand_member, f_new, 0
            else:
                trials[i] += 1

        for _ in range(max_iter):
            for i in range(n_src):  # faza employed
                try_improve(i)
            quality = self._quality(fit)  # faza onlooker
            cumulative = np.cumsum(quality)
            for _ in range(n_src):
                i = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))
                try_improve(i)
            worst = int(trials.argmax())  # faza scout (najwyzej jeden na iteracje)
            if trials[worst] > self.limit:
                foods[worst] = self._new_source(problem, rng)
                member[worst] = False
                member[worst, foods[worst]] = True
                fit[worst] = evaluator(foods[worst])
                trials[worst] = 0
        return max_iter

    @staticmethod
    def _quality(fit: np.ndarray) -> np.ndarray:
        span = fit.max() - fit.min()
        if span == 0.0:
            return np.ones_like(fit)
        return (fit.max() - fit) / span + 0.1

    def _neighbor(
        self,
        solution: np.ndarray,
        member: np.ndarray,
        partner_member: np.ndarray,
        adj: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ruch ograniczony grafem: zamiana CH na sasiada z G, preferujac wezly z rozwiazania partnera."""
        solution = solution.copy()
        member = member.copy()
        for _ in range(self.n_swaps):
            for pos in rng.permutation(len(solution)):
                u = solution[pos]
                free = adj[u] & ~member
                guided = np.flatnonzero(free & partner_member)
                candidates = guided if guided.size else np.flatnonzero(free)
                if candidates.size:
                    v = int(candidates[rng.integers(candidates.size)])
                    member[u], member[v] = False, True
                    solution[pos] = v
                    break
        return solution, member
