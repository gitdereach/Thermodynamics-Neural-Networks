# Wyniki pełnego treningu (PyTorch_TANN)

- `TANN_1D_elasto-hypoplastic_mat_executed.ipynb` – notebook wykonany w całości (5000 iteracji, CPU, ok. 2 min).
- `TANN_results_summary.png` – strata treningu oraz porównanie predykcji (granatowe) z danymi referencyjnymi (czerwone) na ścieżce testowej: Δε_pl, energia ψ, naprężenie σ, dyssypacja d (wartości znormalizowane).
- `run_notebook.py` – uruchamia notebook bez Jupytera:

```bash
pip install -r requirements.txt
python results/run_notebook.py PyTorch_TANN/TANN_1D_elasto-hypoplastic_mat.ipynb PyTorch_TANN out.ipynb
```

Końcowa strata treningu: ~1.8e-3.

## Identyfikacja parametrów konstytutywnych (`PyTorch_TANN/identify_params.py`)

```bash
cd PyTorch_TANN && python identify_params.py            # trening + analiza (~2 min CPU)
cd PyTorch_TANN && python identify_params.py --load ../results/tann_weights.pt
```

Model referencyjny (odczytany z danych, reszta dopasowania ~1e-8): ψ = ½·E·ε_el², σ̇ = E(ε̇ − |ε̇|·σ/S), czyli ε̇_pl = |ε̇|·σ/S.

| Parametr | Referencja | TANN |
|---|---|---|
| E (sztywność sprężysta) | 200 000 | 173 800 (mediana ∂²ψ/∂ε²; zakres 5–95%: 155 500–214 700) |
| S (naprężenie graniczne) | 141.42 | 140.27 (z prawa płynięcia) |
| Rp0.05 | 109.1 | 118.0 |
| Rp0.2 | 138.3 | 137.8 |
| σ przy ε = 0.6% | 141.39 | 138.26 |

Względny błąd RMSE (zbiór testowy, 1 krok): Δε_pl 1.8%, ψ 0.9%, σ 1.8%, D 0.9%.
Wykres: `identified_params.png`; pełne dane: `identified_params.json`; wagi: `tann_weights.pt`.
