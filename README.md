# AE-LSTM-ATT: Attention-Enhanced LSTM Autoencoder for Robotic Anomaly Detection

Code and results supporting the manuscript *"Development and Analysis of
AI-Driven Anomaly Detection and Predictive Maintenance Algorithms for
Robotic Systems in Industrial Environments: Leveraging Electrical and
Sensor Data."*

## Contents

- `ae_lstm_att.py` — Model definitions: `Attention` (Bahdanau-style additive
  attention), `Encoder`, `Decoder`, and the combined `AE_LSTM_ATT` model
  (an autoregressive sequence-to-sequence LSTM autoencoder with attention).
- `Dataset.py` — Dataset wrapper and synthetic anomaly injection logic
  (`spike`, `step`, `freeze_zero`, `freeze_last_value` corruption types).
- `multiseed_eval.py` — Full multi-seed evaluation script: trains
  AE-LSTM-ATT, a matched attention-free LSTM Autoencoder baseline, and an
  Isolation Forest baseline, across 5 random seeds (42, 7, 13, 99, 256),
  with per-anomaly-type F1-optimized detection thresholds tuned on a
  validation split and applied to a held-out test split.
- `results_meanCI.csv` — Final results: mean and 95% confidence interval
  (across the 5 seeds) for AUC-ROC, F1, Precision, and Recall, broken out
  by model and anomaly type.

## Dataset

The IndRAD (Industrial Robot Anomaly Detection) dataset used for training
and evaluation is hosted separately on Kaggle:
`train_dataset_nominal.csv` — 21 columns (`position_0..6`, `velocity_0..6`,
`effort_0..6`) of nominal (fault-free) robot joint sensor readings.

## Running

```bash
export INDRAD_DATA_PATH=/path/to/train_dataset_nominal.csv
python multiseed_eval.py
```

Requires: `torch`, `pandas`, `numpy`, `scikit-learn`, `scipy`.

The script auto-detects a Colab environment (mounting Google Drive for
checkpoint persistence) and otherwise saves checkpoints locally to
`results_multiseed/`. It resumes automatically from the last completed
seed if interrupted.

## Key finding

Across five independent training runs, the attention mechanism's benefit
is anomaly-type dependent rather than uniform:

- On abrupt anomalies (**spike**, **step**), AE-LSTM-ATT and the
  attention-free LSTM Autoencoder baseline perform comparably — both reach
  AUC-ROC = 1.000, with no consistent F1 advantage for attention.
- On subtler, gradually-manifesting anomalies (**freeze-to-zero**,
  **freeze-to-last-value**), AE-LSTM-ATT shows a small, consistent
  advantage over the baseline in both AUC-ROC and F1, most clearly for
  freeze-to-last-value (AUC-ROC: 0.518 ± 0.059 vs. 0.491 ± 0.046).

See `results_meanCI.csv` for full numbers.

## License

Add your preferred license here (e.g., MIT) before publishing.
