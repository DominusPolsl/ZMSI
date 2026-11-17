# Metaheurystyki dla wyboru Cluster Head w WSN

Etap 1 projektu (Zastosowanie metod SI w problemach inżynierskich): dwa dyskretne algorytmy
metaheurystyczne (**Ant Colony System**, **Graph Constrained Bee Colony**) i tester zgodny z
`Wytyczne/Instrukcja-test-podstawa.pdf`.

## Problem

Wybór `k` Cluster Headów spośród `n = 200` węzłów WSN na mapie 1000×1000 m, bez routingu.

- `d0 = sqrt(eps_fs / eps_mp) ≈ 87,7 m` (maks. odległość nadawania w trybie free-space),
- `k = 60` = liczba dysków o promieniu `d0` w pokryciu heksagonalnym mapy (`hex_cover`),
- funkcja celu (minimalizacja): energia nadawania członków do CH + odbiór w CH podzielony przez indeks równości
  obciążeń CH `J` (Jain, niżej) + kara za przeciążone CH (ponad `mediana + IQR`, niżej) + kara za węzły dalej niż `d0`
  od swojego CH. Brak łącza CH→BS.
- model radiowy: Natarajan i in., Sensors 2022 (Eq. 1–3); stałe radiowe: Heinzelman i in., HICSS 2000.

Instancje: `main` (n=200, mapa 1000 m) i `tiny` (n=20, mapa 300 m), na której brute-force daje prawdziwe optimum.

### Bilans obciążenia (indeks Jaina)

```
J    = (Σ_j E_CH,j)² / (k · Σ_j E_CH,j²)
próg = mediana(E_CH) + IQR(E_CH),      IQR = Q3 − Q1
m    = #{ j : E_CH,j > próg },          x = m / k          (część CH przeciążonych)
f    = Σ_i E_TX(d_i) + (Σ_j E_CH,j) / J + (−3·ln x) · Σ_j max(0, E_CH,j − próg) + kara · violation
```
(`−3·ln x = 3·ln(k/m)`; dla `m = 0` człon wynosi 0.)

- `E_CH,j = m_j · l · E_elec` (energia odbioru `m_j` pakietów w CH `j`), `J ∈ [1/k, 1]`, `J = 1` dla równych obciążeń.
  Mnożnik `1/J ∈ [1, k]` powiększa koszt odbioru w CH tym bardziej, im mniej równe są obciążenia. `J` liczymy po wszystkich
  `k` CH, więc CH bez członków obniża `J`.
- Źródło miary: R. Jain, D. Chiu, W. Hawe, *A Quantitative Measure of Fairness and Discrimination for Resource Allocation
  in Shared Computer Systems*, DEC Technical Report TR-301, 1984. Zastosowanie jej do bilansu CH jest naszą decyzją modelową:
  artykuł (Natarajan i in.) takiego członu nie zawiera.
- Przy stałym `k` suma `Σ E_CH = (n−k)·l·E_elec` jest stała, więc człon jest równoważny `k · Σ E_CH² / Σ E_CH`, czyli
  kwadratowej karze za nierówne obciążenia (wariancji obciążeń). Mierzy **ogólną równość**, a nie wąskie gardło
  (CH, który wyczerpie się pierwszy).
- Zastąpił człon `w · max E_CH` z wagą kalibrowaną na referencji zachłannej: siła tamtego członu zależała od przypadkowego
  `max` referencji i nie różnicował rozwiązań (w najlepszych rozwiązaniach ACS i GC-BC dla największego budżetu siatki
  `max` wynosił 5, jak w referencji). Powiązanie wagi z żywotnością sieci wymagałoby założenia, ile ma ona trwać, a tego nie znamy.
- Drugi człon, `(−3·ln x) · Σ_j max(0, E_CH,j − (mediana + IQR))` (w dżulach), karze nadwyżkę obciążenia każdego CH ponad
  próg odporny na outliery. Dodany, bo Jain nie pilnuje pojedynczego, bardzo przeciążonego CH (przy samym Jainie w najlepszych
  rozwiązaniach GC-BC zdarzały się klastry o 6 i 7 członkach). Kwartyle liczymy po wszystkich `k` CH, także pustych
  (interpolacja liniowa), a próg nie ma współczynnika do dobrania. Wariant jest nasz, w duchu reguły wartości odstających
  Tukeya (`Q3 + 1,5·IQR`; Tukey, *Exploratory Data Analysis*, 1977), a nie metoda z literatury o WSN. W naszym modelu
  `E_CH,j` jest proporcjonalne do liczby członków CH (brak agregacji i łącza do stacji bazowej).
- Próg jest względny wobec rozkładu obciążeń danego rozwiązania: szerszy rozrzut podnosi IQR, a więc i próg. Sam rozrzut
  karze indeks Jaina.
- Waga `−3·ln x = 3·ln(k/m)` rośnie, gdy przeciążonych CH jest mało: pojedynczy odstający CH jest karany najmocniej
  (`m = 1`, `k = 60`: waga 12,3; `m = 30`: 2,1). Jeden członek ponad próg kosztuje wtedy ok. 0,0025 J zamiast 0,0002 J przy
  wadze 1, czyli więcej niż rozrzut `f` między uruchomieniami (0,0005 do 0,0009 J). Gdy `m = 0`, człon wynosi 0 (`−ln 0 = ∞`,
  ale nadwyżka też jest zerowa). Ponieważ próg nie jest niższy niż mediana, `x ≤ 0,5`, więc waga dla `m > 0` nie spada poniżej
  2,08, a warunek „waga ≥ 1" byłby zawsze spełniony i nie jest stosowany. CH liczymy jako przeciążony, gdy przekracza próg o więcej niż
  `1e-9 · max E_CH` (odporność na błędy zaokrągleń). Wykładnik 3 i podstawa logarytmu (ln) są stałymi wybranymi przez nas, a wzór jest
  naszą heurystyką bez źródła w literaturze (interpretacja: `−ln x` to informacja niesiona przez zdarzenie o częstości `x`, im
  rzadszy odstający CH, tym ciężej karany).
- Własności i ograniczenia tego członu (pomiar na instancji `main`, N=80, I=80, R=10, porównanie z wersjami „sam Jain" i „waga 1"):
  - Próg jest względny, więc `m = 0` oznacza „żaden CH nie wystaje ponad własny rozkład", a nie brak przeciążenia w liczbach
    bezwzględnych (największy klaster nadal ma 5 członków).
  - Człon przeważa nad `J`. Przejście z wagi 1 na `3·ln(k/m)` kosztuje w składniku Jaina 0,0007 do 0,001 J, a oszczędza w karze
    za przeciążenie 0,004 do 0,005 J, więc algorytmy rozszerzają środek rozkładu (próg rośnie z 4 do 5 członków), żeby wyzerować
    człon. W stosunku do samego Jaina `J` spadł o ok. 0,03 (ACS: 0,783 → 0,751). Funkcja celu daje więc pierwszeństwo
    członowi przeciążenia przed ogólną równością obciążeń.
  - Skraje są krótsze: u GC-BC klaster o co najmniej 6 członkach wystąpił w 1 z 10 uruchomień zamiast w 3 z 10 przy samym Jainie
    (klastry o 7 i 8 członkach zniknęły), a ranking ACS lepszy od GC-BC się utrzymał.
  - Skok wagi przy `m = 0 ↔ m ≥ 1` zwiększył rozrzut wyników przy małym budżecie (N=40, I=40, std ACS z 0,00046 do 0,00120
    względem wagi 1).
- Waga kary `penalty` jest nadal kalibrowana jako `f(referencji)`.

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
uv run python -m project.tester.run_experiments --stage baseline --algo random --workers 4  # baseline losowy
uv run python -m project.tester.run_experiments --stage summarize --out results/summary_all \
    --inputs results/acs_baseline_main results/acs_sensitivity_main \
             results/gcbc_baseline_main results/gcbc_sensitivity_main \
             results/random_baseline_main      # wspólne best-known + wybór najlepszych
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
| `meta.json` | wersje, seedy, siatki, opis instancji, kalibrowana kara (`penalty`), indeks Jaina i nadwyżka obciążenia referencji (`jain_reference`, `overload_reference`) |

Seed uruchomienia `r` to `BASE_SEED + r` (`tester/config.py`), taki sam dla wszystkich konfiguracji.

## Odstępstwa od instrukcji testowania

- Funkcje Sphere, Rosenbrock itd. nie dotyczą algorytmów dyskretnych. Benchmarkiem jest problem CH.
- Optimum instancji `main` jest nieznane, więc zamiast błędu względem optimum raportujemy **gap do best-known**.
  Prawdziwe optimum (brute-force) jest znane tylko dla `tiny`.
- CV = s / |średnia| · 100 % jest tu sensowne, bo f > 0 i średnia jest wyraźnie różna od zera.
