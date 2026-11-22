# Wyniki pełnego treningu (PyTorch_TANN)

- `TANN_1D_elasto-hypoplastic_mat_executed.ipynb` – notebook wykonany w całości (5000 iteracji, CPU, ok. 2 min).
- `TANN_results_summary.png` – strata treningu oraz porównanie predykcji (granatowe) z danymi referencyjnymi (czerwone) na ścieżce testowej: Δε_pl, energia ψ, naprężenie σ, dyssypacja d (wartości znormalizowane).
- `run_notebook.py` – uruchamia notebook bez Jupytera:

```bash
pip install -r requirements.txt
python results/run_notebook.py PyTorch_TANN/TANN_1D_elasto-hypoplastic_mat.ipynb PyTorch_TANN out.ipynb
```

Końcowa strata treningu: ~1.8e-3.
