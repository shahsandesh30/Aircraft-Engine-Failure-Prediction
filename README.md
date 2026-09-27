# Predicting Aircraft Engine Failure Before It Happens

PRT565 — Machine Learning, Artificial Intelligence and Algorithms (Assessment 3).

This project predicts whether a turbofan engine will fail within the **next 30 operating cycles**, using NASA’s C-MAPSS FD001 simulation data.

## Problem

Unplanned engine failure grounds an aircraft and is far more costly than an unnecessary inspection. The notebook frames predictive maintenance as **binary classification** (`y = 1` if remaining useful life ≤ 30 cycles) and reports recall, precision, F1, ROC-AUC and PR-AUC — not accuracy alone.

## Dataset

[NASA C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) subset **FD001**:

- **100 training engines** run to failure (20,631 cycles)
- **100 test engines** stopped before failure, with NASA remaining-life labels
- One operating condition and one fault mode (high-pressure compressor degradation)
- Each row is one cycle: engine id, cycle count, 3 settings, 21 sensors

The notebook downloads the data with `kagglehub` if it is not already in a local `data` folder.

## Approach

- Drop constant sensors and unused operational settings
- Engineer rolling mean/std and per-engine drift features for row-based models
- Split **by engine** (75 train / 25 validation from the run-to-failure set; NASA’s 100 engines as the final test set) to avoid leakage across neighbouring cycles
- Compare a majority-class baseline against Decision Tree, Random Forest, Gaussian Naïve Bayes, MLP, and LSTM
- Score every model on the same rows (cycles ≥ 30)

On the NASA test set, **Random Forest** has the strongest F1 and precision; **LSTM** has the strongest recall and catches all 25 engines that were within 30 cycles of failure at the last observed cycle.

## How to run

The full analysis lives in `code/PRT565_A3_Engine_Failure_Prediction.ipynb`. It runs in **Google Colab** as-is, or locally:

```bash
cd code
python -m venv .venv
.venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Then open the notebook in Jupyter or VS Code / Cursor and run all cells. Random seeds are fixed so results should match a re-run. Figures are written to `code/figures/`.
