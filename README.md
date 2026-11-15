# Metaheurystyki dla wyboru Cluster Head w WSN

Etap 1 projektu (Zastosowanie metod SI w problemach inżynierskich): dwa dyskretne algorytmy
metaheurystyczne (**Ant Colony System**, **Graph Constrained Bee Colony**) i tester zgodny z
`Wytyczne/Instrukcja-test-podstawa.pdf`.

## Problem

Wybór `k` Cluster Headów spośród `n = 200` węzłów WSN na mapie 1000×1000 m, bez routingu.

- `d0 = sqrt(eps_fs / eps_mp) ≈ 87,7 m` (maks. odległość nadawania w trybie free-space),
- `k = 60` = liczba dysków o promieniu `d0` w pokryciu heksagonalnym mapy (`hex_cover`),
- funkcja celu (minimalizacja): energia nadawania członków do CH + odbiór w CH
  + `w · max E_CH` (bilans obciążenia) + kara za węzły dalej niż `d0` od swojego CH. Brak łącza CH→BS.
- model radiowy: Natarajan i in., Sensors 2022 (Eq. 1–3); stałe radiowe: Heinzelman i in., HICSS 2000.

Instancje: `main` (n=200, mapa 1000 m) i `tiny` (n=20, mapa 300 m), na której brute-force daje prawdziwe optimum.

## Algorytmy

- `AntColonySystem` (`algorithms/acs.py`): ACS Dorigo i Gambardella 1997, feromon na węzłach.
- `GraphConstrainedBeeColony` (`algorithms/gcbc.py`): dyskretny ABC z ruchem ograniczonym grafem komunikacji
  (krawędź ⇔ odległość ≤ `radius · d0`). **Nazwa jest robocza: algorytm o nazwie „Graph Constrained Bee Colony"
  nie występuje w literaturze.** Struktura i ruch pochodzą z opublikowanych prac (niżej), a własnym elementem
  jest ograniczenie grafem. Lista odstępstw od źródeł: sekcja „GC-BC: źródła i odstępstwa".
- `RandomSearch`: baseline do sprawdzania, czy metaheurystyki cokolwiek dają.

Wszystkie implementują `core.algorithm.Algorithm` (`run(problem, N, I, seed, max_nfe)`), a NFE i best-so-far
liczy `core.evaluator.Evaluator`.

## GC-BC: źródła i odstępstwa

**Źródła.**
- Karaboga, Basturk, *J. Global Optim.* 39, 2007, 459–471: ABC (pszczoły zatrudnione, obserwatorki, zwiadowcy,
  `limit`, zachłanna selekcja).
- Singh, *Appl. Soft Comput.* 9(2), 2009, 625–631 (**ABC-LCMST**): ABC dla dyskretnego problemu wyboru podzbioru
  węzłów (kodowanie zbiorem węzłów wewnętrznych, każdy liść do najbliższego z nich). Ruch sąsiedni: *usuń losowy węzeł
  i wstaw węzeł z innego, losowo wybranego rozwiązania*. LCMST jest tam opisany jako rozszerzenie problemu p-median,
  a autor wskazuje przeniesienie ruchu na inne problemy wyboru podzbioru.

Ruch w `gcbc.py` (`_neighbor`) jest ruchem Singha. Przy `radius = inf` (graf pełny) zgadza się z nim z dokładnością
do obsługi kolizji (wiersz 2 tabeli).

**Odstępstwa od Singha** (S = Singh 2009, s. 626–629):

| # | Element | Singh (S) | GC-BC | Powód / status |
|---|---|---|---|---|
| 1 | Zakres ruchu | wstawiany węzeł dowolny | wstawiany węzeł musi być **sąsiadem usuwanego w grafie** (`radius · d0`); usuwany CH to pierwszy z losowej permutacji, który ma kandydata | **nasze rozszerzenie** (`radius = inf` je wyłącza); porównanie w sensitivity: `radius ∈ {1, 1.5, 2, inf}` |
| 2 | Brak węzła z partnera (identyczne rozwiązania) | **kolizja**: zatrudniona → scout z nowym losowym rozwiązaniem, obserwatorka → losowanie innego partnera | zamiana na dowolnego sąsiada w grafie spoza rozwiązania, bez kolizji | uproszczenie, bez uzasadnienia merytorycznego |
| 3 | Start | losowe rozwiązania | **losowa konstrukcja zachłanna** (`eta^2`, jak w ACS); `init="random"` zostaje jako ablacja | start losowy nie osiąga rozwiązań dopuszczalnych nawet przy N=80, I=80 (instancja `main`) |
| 4 | Wybór obserwatorek | ruletka, `p_i ∝ 1/W(T_i)` | ruletka na `(f_max − f)/(f_max − f_min) + 0,1` | `1/f` jest prawie jednostajne przy f ≈ 0,1 |
| 5 | Akceptacja ruchu obserwatorki | najlepszy sąsiad spośród obserwatorek jednego źródła razem ze źródłem zastępuje to źródło | każda obserwatorka porównuje swojego sąsiada ze źródłem osobno i od razu je zastępuje | uproszczenie |
| 6 | Zwiadowcy | brak limitu liczby na iterację | **najwyżej jeden** na iterację | Singh zaznacza, że to odstępstwo od innych ABC, w których taki limit jest |
| 7 | `limit` | `2n` iteracji (ok. 400 dla n = 200) | parametr, domyślnie 20, zakres w sensitivity | siatka I ≤ 80 iteracji, więc `limit = 2n` nigdy by nie zadziałał |
| 8 | Dekodowanie rozwiązania | MST na węzłach wewnętrznych + liście do najbliższego | tylko członkowie do najbliższego CH | w naszym problemie nie ma routingu, więc nie ma MST między CH |
| 9 | Ewaluacja | przyrostowa (zmiana jednego węzła) | pełna `evaluate` przy każdym ruchu | wpływa tylko na czas, nie na wyniki ani NFE |
| 10 | Kolonia i stop | 100 pszczół (50 + 50), stop po `4n` iteracjach bez poprawy | N źródeł = N zatrudnionych = N obserwatorek, stała liczba iteracji I | siatka N × I z instrukcji testowania |

Wiersze 2 i 5 są tym, co można by dopasować do Singha bez zmiany idei. W tej wersji pozostają jako odstępstwa.

## Uruchamianie

```bash
uv sync
uv run python -m project.tester.run_experiments --stage verify          # Krok 1: weryfikacja implementacji
uv run python -m project.tester.run_experiments --stage baseline --workers 4     # zestaw bazowy x 4 N x 6 I x R=10
uv run python -m project.tester.run_experiments --stage sensitivity --workers 4  # zestawy OAT x 4 N x 6 I x R=10
uv run python -m project.tester.run_experiments --stage summarize \
    --inputs results/acs_baseline_main results/acs_sensitivity_main \
             results/gcbc_baseline_main results/gcbc_sensitivity_main      # wspólne best-known + wybór najlepszych
uv run python -m project.plots                                                  # wykresy do results/figures
```

Szybki przebieg kontrolny: `--runs 3 --n-values 10,20 --i-values 5,10`.
Opcjonalnie `--max-nfe` ustawia wspólny budżet NFE (porównania przy równym koszcie).

## Wyniki (`results/<algo>_<stage>_<instancja>/`)

| Plik | Zawartość |
|---|---|
| `runs.csv` | jedno uruchomienie = jeden wiersz: parametry, seed, NFE, najlepsze f, najlepszy zbiór CH, czas |
| `summary.csv` | jedna konfiguracja = jeden wiersz: f_min, f_max, średnia, mediana, s (R−1), CV, kwartyle, gap, NFE |
| `histories.jsonl` | krzywe zbieżności f_best(NFE) (tylko zdarzenia poprawy) |
| `meta.json` | wersje, seedy, siatki, opis instancji i wartości kalibrowane (`w_balance`, `penalty`) |

Seed uruchomienia `r` to `BASE_SEED + r` (`tester/config.py`), taki sam dla wszystkich konfiguracji.

## Odstępstwa od instrukcji testowania

- Funkcje Sphere, Rosenbrock itd. nie dotyczą algorytmów dyskretnych. Benchmarkiem jest problem CH.
- Optimum instancji `main` jest nieznane, więc zamiast błędu względem optimum raportujemy **gap do best-known**.
  Prawdziwe optimum (brute-force) jest znane tylko dla `tiny`.
- CV = s / |średnia| · 100 % jest tu sensowne, bo f > 0 i średnia jest wyraźnie różna od zera.
