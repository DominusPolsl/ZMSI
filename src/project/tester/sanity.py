"""Krok 1 instrukcji: weryfikacja implementacji przed wlasciwymi eksperymentami.

Zwykle skrypty z `assert`, bez frameworka testowego. Uruchomienie:
    uv run python -m project.tester.run_experiments --stage verify
"""

from __future__ import annotations

import math
import statistics
import traceback

import numpy as np

from project.algorithms.acs import AntColonySystem
from project.algorithms.gcbc import GraphConstrainedBeeColony
from project.algorithms.random_search import RandomSearch
from project.core.evaluator import BudgetExhausted, Evaluator
from project.problems.wsn_ch import RadioModel, WSNClusterHeadProblem, brute_force, hex_cover
from project.tester.runner import exact_optimum, get_problem
from project.tester.stats import describe_values


def check_objective_by_hand() -> None:
    """Funkcja celu na 4 wezlach zgodna z niezalezna reczna implementacja."""
    radio = RadioModel()
    pos = np.array([[0.0, 0.0], [30.0, 0.0], [200.0, 0.0], [230.0, 0.0]])
    pen = 10.0
    problem = WSNClusterHeadProblem(pos, k=2, side=300.0, radio=radio, penalty=pen)
    l, d0 = radio.packet_bits, radio.d0

    def tx(d: float) -> float:
        amp = radio.eps_fs * d**2 if d <= d0 else radio.eps_mp * d**4
        return l * (radio.e_elec + amp)

    def by_hand(chs: list[int]) -> float:
        members = {c: [] for c in chs}
        total_tx, violation = 0.0, 0.0
        for i in range(4):
            if i in chs:
                continue
            dists = {c: math.dist(pos[i], pos[c]) for c in chs}
            c = min(dists, key=dists.get)
            members[c].append(i)
            total_tx += tx(dists[c])
            if dists[c] > d0:
                violation += (dists[c] - d0) / d0 + 1
        e_ch = [len(m) * l * radio.e_elec for m in members.values()]
        jain = sum(e_ch) ** 2 / (len(chs) * sum(x * x for x in e_ch))  # Jain 1984, po wszystkich k CH
        q1, _, q3 = statistics.quantiles(e_ch, n=4, method="inclusive")  # = interpolacja liniowa numpy
        threshold = statistics.median(e_ch) + (q3 - q1)
        over = [x - threshold for x in e_ch if x - threshold > 1e-9 * max(e_ch)]  # nadwyzki CH ponad progiem
        overload = 3.0 * math.log(len(e_ch) / len(over)) * sum(over) if over else 0.0  # (-3 ln x) * nadwyzka, x = m/k
        return total_tx + sum(e_ch) / jain + overload + pen * violation

    for chs in ([0, 2], [0, 1], [1, 3], [0, 3]):
        got, want = problem.evaluate(np.array(chs)), by_hand(chs)
        assert math.isclose(got, want, rel_tol=1e-12), (chs, got, want)
    assert problem.is_feasible(np.array([0, 2])) and not problem.is_feasible(np.array([0, 1]))


def check_jain_factor() -> None:
    """Indeks Jaina: 1 dla rownych obciazen, 1/k gdy caly ruch w jednym CH, zawsze w [1/k, 1]."""
    problem = get_problem("main")
    k = problem.k
    assert math.isclose(problem._jain(np.full(k, 3.0)), 1.0)
    one = np.zeros(k)
    one[0] = 7.0
    assert math.isclose(problem._jain(one), 1.0 / k)
    assert problem._jain(np.zeros(k)) == 1.0  # n == k: brak czlonkow, mnoznik neutralny
    rng = np.random.default_rng(0)
    for _ in range(50):
        e = rng.random(k) * rng.integers(0, 2, k)  # czesc CH pusta
        if e.sum() == 0.0:
            continue
        j = problem._jain(e)
        assert 1.0 / k - 1e-12 <= j <= 1.0 + 1e-12
        # mnoznik 1/J na sumie E_CH rowna sie k * sum(E^2) / sum(E)
        assert math.isclose(e.sum() / j, k * (e**2).sum() / e.sum(), rel_tol=1e-12)


def check_overload_term() -> None:
    """Kara (-3 ln x) * nadwyzka ponad (mediana + IQR), x = m/k: 0 gdy nic nie przekracza progu, nieujemna, skonczona."""
    problem = get_problem("main")
    lE = problem.radio.packet_bits * problem.radio.e_elec
    assert math.isclose(problem._overload(np.full(problem.k, 3.0 * lE)), 0.0, abs_tol=1e-18)
    # k = 5: mediana = Q1 = Q3 = 1 (IQR 0), prog 1, jeden CH (10) ponad progiem, nadwyzka 9: 3 ln(5/1) * 9
    assert math.isclose(problem._overload(lE * np.array([1.0, 1.0, 1.0, 1.0, 10.0])), 3.0 * math.log(5.0) * 9.0 * lE, rel_tol=1e-12)
    # wiecej niz polowa CH pusta: prog 0, jeden CH (5) ponad progiem: 3 ln(5/1) * 5
    assert math.isclose(problem._overload(lE * np.array([0.0, 0.0, 0.0, 0.0, 5.0])), 3.0 * math.log(5.0) * 5.0 * lE, rel_tol=1e-12)
    # CH dokladnie na progu nie jest przeciazony (odpornosc na bledy zaokraglen): [1..5] ma prog 3 + 2 = 5
    assert math.isclose(problem._overload(lE * np.array([1.0, 2.0, 3.0, 4.0, 5.0])), 0.0, abs_tol=1e-18)
    # k = 60: 59 CH po 2 i jeden 3 (mediana = Q1 = Q3 = 2): jeden przeciazony, nadwyzka 1: 3 ln(60) * 1
    one = lE * np.array([2.0] * 59 + [4.0])  # jeden CH, nadwyzka 2
    assert math.isclose(problem._overload(lE * np.array([2.0] * 59 + [3.0])), 3.0 * math.log(60.0) * lE, rel_tol=1e-12)
    assert math.isclose(problem._overload(one), 3.0 * math.log(60.0) * 2.0 * lE, rel_tol=1e-12)
    # ta sama nadwyzka (2) rozlozona na dwa CH jest karana slabiej: waga 3 ln(60/2) < 3 ln(60/1)
    two = lE * np.array([2.0] * 58 + [3.0, 3.0])
    assert math.isclose(problem._overload(two), 3.0 * math.log(30.0) * 2.0 * lE, rel_tol=1e-12)
    assert problem._overload(one) > problem._overload(two)
    rng = np.random.default_rng(0)
    for _ in range(50):
        e = rng.random(problem.k) * rng.integers(0, 2, problem.k)  # czesc CH pusta
        assert problem._overload(e) >= 0.0 and math.isfinite(problem._overload(e))


def check_k_and_d0() -> None:
    d0 = RadioModel().d0
    assert math.isclose(d0, math.sqrt(10e-12 / 0.0013e-12))
    k_main, centers = hex_cover(1000.0, d0)
    assert k_main == get_problem("main").k == len(centers)
    # pokrycie: kazdy punkt kwadratu lezy w dysku d0 od jakiegos srodka
    g = np.linspace(0, 1000, 120)
    pts = np.array([(x, y) for x in g for y in g])
    nearest = np.sqrt(((pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1)).min(axis=1)
    assert nearest.max() <= d0 + 1e-9, nearest.max()
    # dolne ograniczenie z gestosci pokrycia dyskami (2*pi/(3*sqrt 3) ~ 1.2092)
    assert k_main >= 1.2092 * 1000.0**2 / (math.pi * d0**2) - 1e-9


def check_evaluator_nfe() -> None:
    problem = get_problem("tiny")
    ev = Evaluator(problem, max_nfe=5)
    rng = np.random.default_rng(0)
    prev = math.inf
    for call in range(1, 6):
        ev(problem.random_solution(rng))
        assert ev.nfe == call
        assert ev.best_f <= prev  # best-so-far nie rosnie
        prev = ev.best_f
    assert [h[0] for h in ev.history] == sorted(h[0] for h in ev.history)
    assert all(a[1] > b[1] for a, b in zip(ev.history, ev.history[1:])) and ev.history[-1][1] == ev.best_f
    try:
        ev(problem.random_solution(rng))
    except BudgetExhausted:
        pass
    else:
        raise AssertionError("max_nfe nie zostal wyegzekwowany")
    assert ev.nfe == 5
    ev2 = Evaluator(problem)
    ev2(problem.reference_solution(), track=False)
    assert ev2.nfe == 1 and ev2.best_solution is None  # ewaluacja pomocnicza nie jest wynikiem


def check_solutions_valid_and_nfe() -> None:
    problem = get_problem("main")
    n_agents, iters = 10, 5
    for algo, expected in (
        (RandomSearch(), (n_agents * iters, n_agents * iters)),
        (AntColonySystem(), (n_agents * iters + 1,) * 2),
        (GraphConstrainedBeeColony(), (n_agents + 2 * n_agents * iters, n_agents + 2 * n_agents * iters + iters)),
    ):
        res = algo.run(problem, n_agents, iters, seed=1)
        sol = res.best_solution
        assert len(sol) == len(set(sol)) == problem.k and all(0 <= i < problem.n for i in sol), algo.name
        assert expected[0] <= res.nfe <= expected[1], (algo.name, res.nfe, expected)
        assert math.isclose(problem.evaluate(np.array(sol)), res.best_f, rel_tol=1e-12)
        assert res.history[-1][1] == res.best_f and res.iterations == iters


def check_budget_limit() -> None:
    res = AntColonySystem().run(get_problem("main"), 20, 40, seed=1, max_nfe=150)
    assert res.nfe == 150


def check_reproducibility() -> None:
    problem = get_problem("main")
    for algo in (AntColonySystem(), GraphConstrainedBeeColony(), RandomSearch()):
        a = algo.run(problem, 10, 10, seed=42)
        b = algo.run(problem, 10, 10, seed=42)
        c = algo.run(problem, 10, 10, seed=43)
        assert (a.best_f, a.best_solution, a.history) == (b.best_f, b.best_solution, b.history), algo.name
        assert a.best_solution != c.best_solution or a.best_f != c.best_f, algo.name


def check_feasible_region_exists() -> None:
    main = get_problem("main")
    ref = main.reference_solution()
    assert main.is_feasible(ref) and len(set(ref.tolist())) == main.k
    tiny = get_problem("tiny")
    f_opt, sol = brute_force(tiny)
    assert tiny.is_feasible(sol) and f_opt <= tiny.evaluate(tiny.reference_solution())


def check_statistics() -> None:
    s = describe_values([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(s["f_std"], math.sqrt(5.0 / 3.0)) and s["f_mean"] == 2.5 and s["f_median"] == 2.5
    assert math.isclose(s["f_cv_pct"], s["f_std"] / 2.5 * 100)
    assert s["f_min"] == 1.0 and s["f_max"] == 4.0


def check_tiny_against_brute_force() -> None:
    """Na malej instancji metaheurystyki musza znalezc prawdziwe optimum i bic losowe przeszukiwanie."""
    tiny = get_problem("tiny")
    f_opt, _ = exact_optimum("tiny")  # type: ignore[misc]
    medians = {}
    for algo in (RandomSearch(), AntColonySystem(), GraphConstrainedBeeColony()):
        fs = [algo.run(tiny, 20, 40, seed=1000 + r).best_f for r in range(10)]
        medians[algo.name] = float(np.median(fs))
        assert min(fs) >= f_opt - 1e-12  # nic nie moze byc lepsze od optimum
        if algo.name != "random":
            assert min(fs) <= f_opt * (1 + 1e-9), (algo.name, min(fs), f_opt)
    assert medians["acs"] < medians["random"] and medians["gcbc"] < medians["random"], medians


def check_beats_random_on_main() -> None:
    """Przy tym samym NFE oba algorytmy musza wyraznie bic losowe przeszukiwanie."""
    problem = get_problem("main")
    budget = 1000
    random_f = np.median([RandomSearch().run(problem, 20, 50, seed=s, max_nfe=budget).best_f for s in range(5)])
    for algo in (AntColonySystem(), GraphConstrainedBeeColony()):
        f = np.median([algo.run(problem, 20, 50, seed=s, max_nfe=budget).best_f for s in range(5)])
        assert f < random_f, (algo.name, f, random_f)


CHECKS = [
    check_objective_by_hand,
    check_jain_factor,
    check_overload_term,
    check_k_and_d0,
    check_evaluator_nfe,
    check_solutions_valid_and_nfe,
    check_budget_limit,
    check_reproducibility,
    check_feasible_region_exists,
    check_statistics,
    check_tiny_against_brute_force,
    check_beats_random_on_main,
]


def run_sanity() -> bool:
    ok = True
    for check in CHECKS:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except Exception:  # noqa: BLE001 - chcemy zobaczyc kazda porazke
            ok = False
            print(f"FAIL  {check.__name__}")
            traceback.print_exc()
    print("Wszystkie testy OK" if ok else "Sa bledy")
    return ok
