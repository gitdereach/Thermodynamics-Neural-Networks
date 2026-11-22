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
