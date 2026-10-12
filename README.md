# Reproducible ML Pipeline · NB4 (UNMSM)

**EN:** Logistic regression on the Breast Cancer Wisconsin (Diagnostic) dataset. Built so anyone can reproduce the exact results.
**ES:** Regresión logística sobre el dataset Breast Cancer Wisconsin (Diagnostic). Diseñado para que cualquiera pueda reproducir exactamente los mismos resultados.

## Dataset & License
- **Source:** UCI Machine Learning Repository / `sklearn.datasets.load_breast_cancer`
- **Target Encoding:** `0` = Malignant, `1` = Benign
- **License:** MIT License (see `LICENSE`)

## Headline Result
- **Across-seed accuracy (5 seeds):** `0.9803 ± 0.0021` (Mean ± SD)
- **Single run example (Seed 42):** `0.97368 ± 0.01292`

## How to reproduce
```bash
git clone [https://github.com/](https://github.com/)<your-username>/repro-lab.git && cd repro-lab
pip install -r requirements.txt

# Single run
PYTHONHASHSEED=0 python src/train.py --seed 42

# Regenerate complete 5-seed summary table
PYTHONHASHSEED=0 python src/run_seeds.py

# Run automated test suite
PYTHONHASHSEED=0 pytest tests/

- **Data Fingerprint (MD5):** `5afc23b9622f8f9ffa60824f4e97d0d5`
- **Python Version:** `3.13.16`
