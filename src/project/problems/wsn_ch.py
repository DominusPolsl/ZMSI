"""Problem wyboru Cluster Head (CH) w sieci WSN, bez routingu.

Zrodla modelu:
  * model radiowy pierwszego rzedu (Eq. 1-3 w Natarajan et al., Sensors 2022, 22, 3761),
  * stale radiowe (E_elec, eps_fs, eps_mp): Heinzelman et al., HICSS 2000 (LEACH),
    bo artykul nie podaje ich wartosci liczbowych (sam model pochodzi z artykulu, bez energii agregacji),
  * rozmiar pakietu 512 B i 200 wezlow na mapie 1000x1000 m: Tabela 1 artykulu.

Liczba CH `k` wynika z pokrycia mapy dyskami o promieniu d0 = sqrt(eps_fs/eps_mp), czyli
z maksymalnej odleglosci, na ktora wezel nadaje w trybie free-space.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from project.core.problem import Construction, SubsetProblem


@dataclass(frozen=True)
class RadioModel:
    e_elec: float = 50e-9  # J/bit, nadajnik i odbiornik
    eps_fs: float = 10e-12  # J/bit/m^2, free space
    eps_mp: float = 0.0013e-12  # J/bit/m^4, multipath
    packet_bits: int = 4096  # 512 B

    @property
    def d0(self) -> float:
        """Eq. 3: odleglosc progowa free-space / multipath."""
        return math.sqrt(self.eps_fs / self.eps_mp)

    def e_tx(self, d: np.ndarray) -> np.ndarray:
        """Eq. 1: energia nadania pakietu na odleglosc d."""
        d = np.asarray(d, dtype=float)
        amp = np.where(d <= self.d0, self.eps_fs * d**2, self.eps_mp * d**4)
        return self.packet_bits * (self.e_elec + amp)


def hex_cover(side: float, r: float, steps: int = 16) -> tuple[int, np.ndarray]:
    """Minimalna liczba dyskow o promieniu r w pokryciu heksagonalnym kwadratu [0, side]^2.

    Siatka heksagonalna (rozstaw w wierszu sqrt(3)*r, rozstaw wierszy 1.5*r) pokrywa cala plaszczyzne
    sesciokatami wpisanymi w dyski. Liczymy srodki, ktorych sesciokat przecina kwadrat
    (test SAT dla wypuklych wielokatow) i minimalizujemy po przesunieciu siatki.
    Zwraca (liczba dyskow, ich srodki).
    """
    a, b = math.sqrt(3.0) * r, 1.5 * r
    ang = np.deg2rad(30.0 + 60.0 * np.arange(6))
    hex_v = r * np.column_stack([np.cos(ang), np.sin(ang)])
    axes = np.array([[1.0, 0.0], [0.0, 1.0]] + [[math.cos(t), math.sin(t)] for t in np.deg2rad([0, 60, 120])])
    corners = np.array([[0, 0], [side, 0], [0, side], [side, side]], dtype=float)
    hex_half = np.abs(hex_v @ axes.T).max(axis=0)
    rect_proj = corners @ axes.T
    rect_lo, rect_hi = rect_proj.min(axis=0), rect_proj.max(axis=0)

    best: np.ndarray | None = None
    for ox in np.linspace(0.0, a, steps, endpoint=False):
        for oy in np.linspace(0.0, b, steps, endpoint=False):
            pts = []
            for j in range(-3, int(side / b) + 4):
                x_shift = a / 2.0 if j % 2 else 0.0
                for i in range(-3, int(side / a) + 4):
                    pts.append((ox + i * a + x_shift, oy + j * b))
            centers = np.array(pts)
            proj = centers @ axes.T
            overlap = ((proj + hex_half) > rect_lo + 1e-9) & ((proj - hex_half) < rect_hi - 1e-9)
            kept = centers[overlap.all(axis=1)]
            if best is None or len(kept) < len(best):
                best = kept
    assert best is not None
    return len(best), best


class _WSNConstruction(Construction):
    """Dynamiczna heurystyka pokrycia: eta_i = suma po jeszcze niepokrytych j w zasiegu d0 z (1 - d_ij/d0)."""

    EPS = 1e-4

    def __init__(self, problem: WSNClusterHeadProblem) -> None:
        self._p = problem
        self._uncovered = np.ones(problem.n, dtype=bool)
        self._eta = problem._weights.sum(axis=1)

    def eta(self) -> np.ndarray:
        return np.maximum(self._eta, 0.0) + self.EPS

    def add(self, node: int) -> None:
        newly = self._uncovered & (self._p.dist[node] <= self._p.d0)
        if newly.any():
            self._eta -= self._p._weights[:, newly].sum(axis=1)
            self._uncovered &= ~newly


class WSNClusterHeadProblem(SubsetProblem):
    """Wybor k CH sposrod n wezlow. Kazdy pozostaly wezel nadaje do najblizszego CH.

    f = sum_i E_TX(d_i) + sum_j E_CH,j + w * max_j E_CH,j + kara * (sum_i max(0, d_i - d0)/d0 + #niepokrytych)

    gdzie E_CH,j = m_j*l*E_elec = m_j*E_RX (Eq. 2), a "niepokryty" to wezel dalej niz d0
    od swojego CH. Brak lacza CH->BS (poza zakresem: routing). Wagi `w` i `kara` sa kalibrowane na
    rozwiazaniu zachlannym (patrz `_calibrate`), chyba ze podano je jawnie.
    """

    def __init__(
        self,
        positions: np.ndarray,
        k: int,
        side: float,
        radio: RadioModel | None = None,
        name: str = "wsn",
        w_balance: float | None = None,
        penalty: float | None = None,
    ) -> None:
        self.name = name
        self.positions = np.asarray(positions, dtype=float)
        self.n = len(self.positions)
        self.k = k
        self.side = side
        self.radio = radio or RadioModel()
        self.d0 = self.radio.d0

        # creating new dimension with None, and then substracting the rotated column from the column with two dimension values, creating the broadcasted square matrix
        diff = self.positions[:, None, :] - self.positions[None, :, :] 

        self.dist = np.sqrt((diff**2).sum(axis=-1))

        # This weights are going to be applied in the heuristic function, they tell wether this node is more attractive
        self._weights = np.clip(1.0 - self.dist / self.d0, 0.0, None)
        self._adj_cache: dict[float, np.ndarray] = {}
        self.w_balance = 0.0
        self.penalty = 0.0
        self._reference = self._greedy_reference()
        self._calibrate(w_balance, penalty)

    # ------------------------------------------------------------------ funkcja celu
    def _terms(self, solution: np.ndarray) -> tuple[float, np.ndarray, float]:
        sol = np.asarray(solution, dtype=np.int64) # chosen CH indexes list
        dsub = self.dist[:, sol] # distance from the node to any other chosen CH
        assign = dsub.argmin(axis=1) # Closest CH index to each node
        dmin = dsub[np.arange(self.n), assign] # Distance from each node to its closest CH
        is_ch = np.zeros(self.n, dtype=bool)
        is_ch[sol] = True
        member = ~is_ch
        m = np.bincount(assign[member], minlength=self.k) # Counts the number of each unique element sorted in the given series, but when minlenght is specified than it takes into account all of the elements for case of arr = np.array([0, 1, 1, 0, 3]) i minlength=4 it is goin to be array([2, 2, 0, 1])
        e_members = float(self.radio.e_tx(dmin[member]).sum())
        bits = self.radio.packet_bits
        e_ch = m * bits * self.radio.e_elec
        excess = np.maximum(dmin[member] - self.d0, 0.0)
        violation = float(excess.sum() / self.d0 + (excess > 0).sum()) # penalty, we count there the excess, which means the commulative sum of all nodes working in multipath mode. The minimum violation is atleast one due to the counting of the nodes being to far from the CH
        return e_members, e_ch, violation

    def evaluate(self, solution: np.ndarray) -> float:
        e_members, e_ch, violation = self._terms(solution)
        return e_members + float(e_ch.sum()) + self.w_balance * float(e_ch.max()) + self.penalty * violation

    def is_feasible(self, solution: np.ndarray) -> bool:
        return self._terms(solution)[2] == 0.0

    # ------------------------------------------------------------------ interfejs SubsetProblem
    # Calculating the adjacency matrix only once for the given radius and than remembering it in cache. Necessary for ABC
    def adjacency(self, scale: float) -> np.ndarray:
        key = float(scale)
        if key not in self._adj_cache:
            eye = np.eye(self.n, dtype=bool)
            if math.isinf(key):
                adj = ~eye
            else:
                adj = (self.dist <= key * self.d0) & ~eye
            self._adj_cache[key] = adj
        return self._adj_cache[key]

    def start_construction(self) -> Construction:
        return _WSNConstruction(self)

    def reference_solution(self) -> np.ndarray:
        return self._reference.copy()

    # ------------------------------------------------------------------ referencja zachlanna i kalibracja
    def _greedy_reference(self) -> np.ndarray:
        """Zachlanne pokrycie zbioru (promien d0), uzupelnione do k zachlannym p-median."""
        covers = self.dist <= self.d0
        uncovered = np.ones(self.n, dtype=bool)
        chosen: list[int] = []
        while uncovered.any() and len(chosen) < self.k:
            gain = (covers & uncovered).sum(axis=1)
            gain[chosen] = -1
            node = int(gain.argmax())  # remis: najnizszy indeks (deterministycznie)
            chosen.append(node)
            uncovered &= ~covers[node]
        dmin = self.dist[:, chosen].min(axis=1)
        while len(chosen) < self.k:
            cost = (np.minimum(dmin[:, None], self.dist) ** 2).sum(axis=0)
            cost[chosen] = np.inf
            node = int(cost.argmin())
            chosen.append(node)
            dmin = np.minimum(dmin, self.dist[:, node])
        return np.array(chosen, dtype=np.int64)

    def _calibrate(self, w_balance: float | None, penalty: float | None) -> None:
        """w: czlon bilansu rowny czlonowi zwartosci (zmienna czesc energii nadawania) na referencji.
        kara: f(referencji), wiec kazde rozwiazanie niedopuszczalne (kara >= f_ref) jest gorsze od referencji."""
        e_members, e_ch, _ = self._terms(self._reference)
        bits = self.radio.packet_bits
        variable = e_members - (self.n - self.k) * bits * self.radio.e_elec
        self.w_balance = float(variable / e_ch.max()) if w_balance is None else float(w_balance)
        if penalty is None:
            self.penalty = 0.0
            self.penalty = self.evaluate(self._reference)
        else:
            self.penalty = float(penalty)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "k": self.k,
            "side_m": self.side,
            "d0_m": self.d0,
            "packet_bits": self.radio.packet_bits,
            "e_elec": self.radio.e_elec,
            "eps_fs": self.radio.eps_fs,
            "eps_mp": self.radio.eps_mp,
            "w_balance": self.w_balance,
            "penalty": self.penalty,
            "reference_f": self.evaluate(self._reference),
            "reference_feasible": self.is_feasible(self._reference),
        }


def brute_force(problem: SubsetProblem) -> tuple[float, np.ndarray]:
    """Pelne przeszukanie wszystkich C(n, k) podzbiorow (tylko dla malej instancji)."""
    best_f, best = math.inf, None
    for comb in itertools.combinations(range(problem.n), problem.k):
        sol = np.fromiter(comb, dtype=np.int64, count=problem.k)
        f = problem.evaluate(sol)
        if f < best_f:
            best_f, best = f, sol
    assert best is not None
    return best_f, best


# ---------------------------------------------------------------------- instancje
def make_instance(n: int, side: float, seed: int, name: str, radio: RadioModel | None = None) -> WSNClusterHeadProblem:
    radio = radio or RadioModel()
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, side, size=(n, 2))
    k, _ = hex_cover(side, radio.d0)
    return WSNClusterHeadProblem(positions, k=k, side=side, radio=radio, name=name)


MAIN_SEED = 2026
TINY_SEED = 7


def main_instance() -> WSNClusterHeadProblem:
    """200 wezlow na mapie 1000x1000 m (Tabela 1 artykulu)."""
    return make_instance(200, 1000.0, MAIN_SEED, "main")


def tiny_instance() -> WSNClusterHeadProblem:
    """Mala instancja weryfikacyjna, na ktorej brute-force daje prawdziwe optimum."""
    return make_instance(20, 300.0, TINY_SEED, "tiny")


INSTANCES = {"main": main_instance, "tiny": tiny_instance}
