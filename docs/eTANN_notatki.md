# TANN z automatycznym odkrywaniem zmiennych wewnętrznych – notatki z literatury

Źródła (przeczytane w całości):

- F. Masi, I. Stefanou (2022), *Multiscale modeling of inelastic materials with Thermodynamics-based Artificial Neural Networks (TANN)*, CMAME 398, 115190 — [arXiv:2108.13137](https://arxiv.org/abs/2108.13137). Kod: [github.com/filippo-masi/TANN-multiscale](https://github.com/filippo-masi/TANN-multiscale) (generator danych MES dla kratownic + gotowy model TANN jako UMAT w Numerical Geolab; **bez kodu autoenkodera/treningu**).
- F. Masi, I. Stefanou (2023), *Evolution TANN and the identification of internal variables and evolution equations in solid mechanics*, JMPS, doi:10.1016/j.jmps.2023.105245 — [arXiv:2209.13269](https://arxiv.org/abs/2209.13269). Kod: brak publicznego; „All data is available on request from F.M.”

## Idea eTANN

Ramy termodynamiczne (izotermicznie, małe odkształcenia):

- ψ = ψ̂(ε, z), σ = ∂ψ̂/∂ε, d = −∂ψ̂/∂z · ż ≥ 0
- prawo ewolucji jako **równanie różniczkowe** ż = f(ε, z) (a nie przyrost dla zadanego kroku jak w TANN 2020) → wynik niezależny od wielkości kroku, możliwa lepkość (zależność od prędkości).

Bloki sieci:

1. **Sieć energii swobodnej** ψ̂(ε, z). Strata: L_energy = L_ψ + L_gradψ + λ_reg·L_reg, gdzie
   L_gradψ = ‖σ − ∂_ε ψ̂‖ + ‖d + ∂_z ψ̂ · ż‖, a L_reg = ⟨∂_z ψ̂ · ż⟩ (nawiasy Macaulaya: kara za ujemną dyssypację).
   Autorzy zauważają, że **L_ψ (sama energia) nie jest konieczne** – wystarczy uczenie przez gradienty.
2. **Sieć prawa ewolucji** f(ε, z), trenowana „derivative fitting”: ‖ż − f(ε, z)‖ (wymaga znajomości ż).
3. **Gdy zmienne wewnętrzne nie są znane – autoenkoder**: z = h(ξ), ξ̃ = g(z), gdzie ξ to **„współrzędne wewnętrzne” = pola mikroskopowe** (np. odkształcenia całkowite i niesprężyste wszystkich prętów kratownicy). ż = ∂_ξ h · ξ̇.
   Strata całkowita: L = L_recon + L_ξ̇ + L_ż + L_ψ + L_gradψ + λ_reg·L_reg (wagi = 1 po normalizacji).
   Wymiar przestrzeni ukrytej n_z dobierany analizą wrażliwości (dla kratownicy 24 prętów: n_z = 33).

Całkowanie w czasie po treningu: zewnętrzny solver ODE (BDF, SciPy).

## Przykłady i skala danych (wszystkie z symulacji)

| Przykład | Zmienne wewnętrzne | Dane |
|---|---|---|
| 3D, von Mises + umocnienie izotropowe | znane (z_pl, z_q) | 40 000 |
| 3D, plastyczność + uszkodzenie | znane (z_pl, z_d) | 40 000 |
| 3D, lepkoplastyczność Perzyny | znane | 60 000 |
| kratownica 24 prętów, lepkoplastyczna | **odkrywane** (n_z = 33) | 56 000 |

Podział 60/20/20 %, Adam (Nesterov), lr 1e-3, mini-batch 1000, TensorFlow.

## Wnioski dla danych eksperymentalnych

- Odkrywanie zmiennych wewnętrznych **wymaga pól mikroskopowych ξ** (z symulacji mikrostruktury). Z testu jednoosiowego (siła–wydłużenie) nie ma czego podać autoenkoderowi.
- Pozostałe straty potrzebują σ i **d** (dyssypacji) – d nie jest mierzalne bez termografii.
- Dla danych ε–σ potrzebne jest więc inne podejście: zmienna wewnętrzna całkowana wzdłuż ścieżki („trajectory fitting”) i strata tylko na σ + kara za ujemną dyssypację. Zob. `PyTorch_TANN/discover_internal_var.py`.

---

# Kolejne prace (przesłane PDF-y)

## Masi, Stefanou, Vannucci, Maffi-Berthier (2021) – *Material modeling via TANN*, SPIGL'20 (Springer)

Plakat/skrót pracy JMPS 2020 (to, co implementuje to repozytorium): F = F̃(ε, Z), σ = ∂F/∂ε, D = −Σ ∂F/∂ζᵢ · ζ̇ᵢ ≥ 0,
D jednorodne stopnia 1 w Ż (niezależność od prędkości). Dodatkowa uwaga praktyczna: **problem zanikających drugich
pochodnych** – klasyczne aktywacje (ReLU) nie nadają się do sieci energii, bo naprężenie i sztywność to jej pochodne
(stąd w kodzie TF aktywacja ELU(x²), a w naszych skryptach softplus/GELU). Porównanie TANN vs zwykła ANN: TANN
stabilniejsze w treningu i zawsze termodynamicznie spójne.

## Masi, Stefanou (2023) – *Evolution TANN…*, JMPS 173, 105245 (wersja opublikowana)

Treść zgodna z preprintem arXiv:2209.13269 opisanym wyżej. Brak publicznego kodu.

## Masi, Einav (2024) – *Neural integration for constitutive equations using small data* (NICE), CMAME 420, 116698

**Najważniejsza praca z punktu widzenia danych eksperymentalnych.** Kod: [github.com/filippo-masi/NICE](https://github.com/filippo-masi/NICE)
(PyTorch + `torchdiffeq`, notebooki z dwoma benchmarkami, dane i wytrenowane wagi).

- Problem: metody przyrostowe (w tym TANN 2020 i eTANN) potrzebują pełnego stanu materiału w każdym kroku (np. ε_pl,
  przyrosty zmiennych) – tego eksperyment nie daje („big problem of small data”).
- Rozwiązanie: **neuronowe równania różniczkowe** (neural ODE). Stan 𝒮 = {ρ, εᵉ, z} całkowany w czasie
  (metoda punktu środkowego, h = T/800) z warunku początkowego; strata liczona na **trajektorii** naprężenia.
- Termodynamika z energią wewnętrzną u(ρ, εᵉ, z): σ = ∂u/∂εᵉ + (ρ ∂u/∂ρ − u)·1, d = σᵉ:(ε̇ − ε̇ᵉ) − τ·ż ≥ 0.
- **Odkształcenie sprężyste nie jest mierzone** – sieć odtwarza je sama; jedynie εᵉ(0) jest uczonym parametrem
  (z warunku σ(0) = σ_zmierzone). Odkryte εᵉ zgadza się z prawdą z dokładnością do stałej.
- Strata: L = L_σ + L_z + L_d + ‖r_εᵉ(0)‖² + λ(‖θ‖² + ‖ω‖²); **bez straty na energii i dyssypacji** (niepotrzebne).
  L_z wymaga zmiennych dyssypacyjnych tylko w M punktach – testowano M = 2 (tylko początek i koniec testu).
- Dane treningowe naśladują **standardowe protokoły laboratoryjne**: monotoniczne obciążenie + odciążenie do zera
  (bez cykli!), N = 40 (benchmark 1) lub 140 (benchmark 2) punktów naprężenia na ścieżkę, podział 60/20/20.
  Mimo to model poprawnie przewiduje cykle odciążenie–ponowne obciążenie i niewidziane ścieżki.
- Benchmarki: (1) Drucker–Prager, K = 70 GPa, G = 60 GPa – błąd σ 0.9%; (2) ośrodek porowaty
  (εᵉ, ρ, frakcja fazy stałej φ), 4 próbki o różnym φ_init.
- **Szum**: do 20% szumu gaussowskiego w danych → błąd predykcji zawsze mniejszy niż poziom szumu (8% przy 20%).
  Sformułowanie przyrostowe (dodatek A) zawodzi przy danych niepełnych lub rzadkich, niezależnie od termodynamiki.
- Ograniczenia: rodzaj zmiennych stanu zakładany z góry; tylko izotermicznie, bez lepkości; brak testu na prawdziwych
  danych eksperymentalnych (wskazane jako cel przyszłych prac).

### Związek z naszym `discover_internal_var.py`

Ta sama idea (całkowanie stanu wzdłuż ścieżki, strata tylko na σ, kara za ujemną dyssypację), różnice:
NICE używa εᵉ jako zmiennej stanu i ciągłego czasu (ODE), my – z przy całkowitym ε i krokach przyrostowych
dz = |dε|·F(ε, z, sgn dε). Oba podejścia odtwarzają zmienną wewnętrzną z dokładnością do stałej/skali.
