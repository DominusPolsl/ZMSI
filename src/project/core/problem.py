"""Abstrakcyjny problem wyboru k-elementowego podzbioru wezlow grafu.

Algorytmy (ACS, GC-BC) znaja tylko ten interfejs, nie wiedza nic o WSN.
Rozwiazanie to tablica numpy z k *roznymi* indeksami wezlow z zakresu [0, n).
Funkcja celu jest minimalizowana.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Construction(ABC):
    """Stan konstrukcji rozwiazania krok po kroku (uzywany przez mrowki)."""

    @abstractmethod
    def eta(self) -> np.ndarray:
        """Informacja heurystyczna eta_i > 0 dla wszystkich n wezlow przy biezacym stanie."""

    @abstractmethod
    def add(self, node: int) -> None:
        """Dodaje wezel do czesciowego rozwiazania i aktualizuje stan."""


class SubsetProblem(ABC):
    """Wybor k z n wezlow grafu, minimalizacja funkcji celu."""

    n: int
    k: int

    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> float:
        """Wartosc funkcji celu dla k roznych indeksow. Nie liczy NFE (robi to Evaluator)."""

    @abstractmethod
    def adjacency(self, scale: float) -> np.ndarray:
        """Macierz sasiedztwa (n, n, bool, bez petli) grafu ograniczajacego ruchy.

        `scale` skaluje promien grafu; np.inf oznacza graf pelny (brak ograniczenia).
        """

    @abstractmethod
    def start_construction(self) -> Construction:
        """Nowy, pusty stan konstrukcji dla jednej mrowki."""

    @abstractmethod
    def reference_solution(self) -> np.ndarray:
        """Deterministyczne rozwiazanie referencyjne (zachlanne), np. do wyznaczenia tau0."""

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.n, size=self.k, replace=False)

    def heuristic_solution(self, rng: np.random.Generator, beta: float = 2.0) -> np.ndarray:
        """Losowa konstrukcja zachlanna: kolejne wezly losowane z prawdopodobienstwem ~ eta_i^beta."""
        construction = self.start_construction()
        chosen = np.zeros(self.n, dtype=bool)
        solution = np.empty(self.k, dtype=np.int64)
        for step in range(self.k):
            score = construction.eta() ** beta
            score[chosen] = 0.0
            cumulative = np.cumsum(score)
            node = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))
            chosen[node] = True
            solution[step] = node
            construction.add(node)
        return solution
