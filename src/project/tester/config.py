"""Konfiguracja eksperymentow: siatki N, I, liczba powtorzen, seedy i zestawy parametrow wewnetrznych.

Parametry ogolne wg Instrukcja-test-podstawa (Krok 2 i 3): N in {10,20,40,80}, I in {5,10,20,40,60,80}, R = 10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from project.algorithms.acs import AntColonySystem
from project.algorithms.gcbc import GraphConstrainedBeeColony
from project.algorithms.random_search import RandomSearch
from project.core.algorithm import Algorithm

N_VALUES = (10, 20, 40, 80)
I_VALUES = (5, 10, 20, 40, 60, 80)
R = 10
BASE_SEED = 1000  # uruchomienie r uzywa seeda BASE_SEED + r (te same seedy dla wszystkich konfiguracji)

ALGORITHMS: dict[str, type[Algorithm]] = {
    "acs": AntColonySystem,
    "gcbc": GraphConstrainedBeeColony,
    "random": RandomSearch,
}


@dataclass(frozen=True)
class ParameterSet:
    """Jeden zestaw parametrow wewnetrznych (P1..Pk) algorytmu."""

    set_id: str
    algo: str
    params: tuple[tuple[str, object], ...]  # (nazwa, wartosc); krotka, zeby obiekt byl hashowalny

    @property
    def as_dict(self) -> dict:
        return dict(self.params)

    def build(self) -> Algorithm:
        return ALGORITHMS[self.algo](**self.as_dict)


def _set(algo: str, set_id: str, **params) -> ParameterSet:
    return ParameterSet(set_id=set_id, algo=algo, params=tuple(params.items()))


def _oat(algo: str, base: dict, variants: dict[str, tuple]) -> list[ParameterSet]:
    """Zestawy 'one-at-a-time': baza + warianty zmieniajace po jednym parametrze."""
    sets = [_set(algo, "base", **base)]
    for name, values in variants.items():
        for value in values:
            if value == base[name]:
                continue
            sets.append(_set(algo, f"{name}={value}", **{**base, name: value}))
    return sets


# ACS: parametry bazowe z Dorigo & Gambardella 1997 (beta=2, rho=0.1, xi=0.1, q0=0.9).
ACS_BASE = dict(beta=2.0, rho=0.1, xi=0.1, q0=0.9)
ACS_VARIANTS = {
    "q0": (0.5, 0.7, 0.9, 0.95),
    "rho": (0.05, 0.1, 0.3),
    "beta": (1.0, 2.0, 5.0),
    "xi": (0.05, 0.1, 0.3),
}

# GC-BC: baza wlasna (brak publikacji zrodlowej). Zakresy limit dobrane do I <= 80
# (klasyczny limit = SN*D jest przy takich I nieosiagalny).
GCBC_BASE = dict(limit=20, n_swaps=1, radius=1.0, init="heuristic")
GCBC_VARIANTS = {
    "limit": (5, 10, 20, 40),
    "n_swaps": (1, 2, 3),
    "radius": (1.0, 1.5, 2.0, math.inf),
    "init": ("heuristic", "random"),
}

PARAMETER_SETS: dict[str, list[ParameterSet]] = {
    "acs": _oat("acs", ACS_BASE, ACS_VARIANTS),
    "gcbc": _oat("gcbc", GCBC_BASE, GCBC_VARIANTS),
    "random": [_set("random", "base")],
}


def select_sets(algo: str, stage: str) -> list[ParameterSet]:
    """stage: baseline = zestaw bazowy, sensitivity = zestawy OAT bez bazowego, all = wszystkie."""
    sets = PARAMETER_SETS[algo]
    if stage == "baseline":
        return [s for s in sets if s.set_id == "base"]
    if stage == "sensitivity":
        return [s for s in sets if s.set_id != "base"]
    if stage == "all":
        return list(sets)
    raise ValueError(f"nieznany stage: {stage}")
