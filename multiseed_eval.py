#!/usr/bin/env python3
"""
Multi-seed evaluation of the REAL AE-LSTM-ATT model (Bahdanau-style additive
attention, autoregressive seq2seq decoder -- see ae_lstm_att.py) against a
plain LSTM-AE baseline and Isolation Forest, across 5 seeds, with per-anomaly-
type F1-optimized thresholds (tuned on validation, applied to test).

This produces the mean +/- 95% CI table needed for the paper's Table 3 and
resolves editor issue #3 (missing confidence intervals) using real numbers.

REQUIRES (place these two files next to this script, or adjust sys.path):
  - ae_lstm_att.py   (your real model file: Attention, Encoder, Decoder, AE_LSTM_ATT)
  - Dataset.py       (your real data/corruption class)

SET the data path via the INDRAD_DATA_PATH environment variable, e.g.:
  export INDRAD_DATA_PATH=/kaggle/input/datasets/salisuauwal/indrad-dataset/train_dataset_nominal.csv
"""
import os, sys, time, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve

sys.path.insert(0, os.getcwd())
from ae_lstm_att import AE_LSTM_ATT
from Dataset import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ---- Colab-specific: mount Google Drive so checkpoints SURVIVE a runtime
# disconnect. Colab's local disk (/content/...) is wiped on disconnect/restart --
# writing checkpoints there only protects against errors mid-session, not
# against Colab kicking you off entirely, which is what happened last time.
# NOTE: google.colab can be importable even on non-Colab platforms (e.g. Kaggle),
# so we must catch failures from drive.mount() itself, not just the import.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    RESULTS_DIR = "/content/drive/MyDrive/ae_lstm_att_results"
    print(f"Running on Colab -- results will be saved to Google Drive: {RESULTS_DIR}")
except Exception:
    RESULTS_DIR = "results_multiseed"
    print(f"Not on Colab (or Drive mount unavailable) -- results saved locally: {RESULTS_DIR}")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_DATA_PATH = "dataset_train/train_dataset_nominal.csv"
DATA_PATH = os.environ.get("INDRAD_DATA_PATH", DEFAULT_DATA_PATH)

# ---- config, matching your real v2 run ----
WINDOW_SIZE   = 50
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 30
PATIENCE      = 3
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
SEEDS         = [42, 7, 13, 99, 256]
ANOMALY_TYPES = ["spike", "step", "freeze_zero", "freeze_last_value"]
INPUT_SIZE    = 21  # position_0..6, velocity_0..6, effort_0..6


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class LSTMAEBaseline(nn.Module):
    """Plain encoder-decoder LSTM, no attention -- structurally matched to
    AE_LSTM_ATT (same encoder, autoregressive decoder) but the decoder does
    NOT receive an attention context vector, isolating attention's effect."""
    def __init__(self, input_size, hidden_size, num_layers, seq_len, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        outputs = torch.zeros(x.size(0), self.seq_len, self.input_size, device=x.device)
        decoder_input = x[:, -1, :].unsqueeze(1)
        for t in range(self.seq_len):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            step_out = self.out(out.squeeze(1))
            outputs[:, t, :] = step_out
            decoder_input = step_out.unsqueeze(1)
        return outputs, None


def make_windows(df, w):
    d = df.values.astype(np.float32)
    idx = np.arange(w)[None, :] + np.arange(len(d) - w + 1)[:, None]
    return d[idx]


def make_loader(data, shuffle=False):
    t = torch.from_numpy(data).float()
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(t), batch_size=BATCH_SIZE, shuffle=shuffle)


def make_anomaly_set(df_clean, corrupt_type, scaler):
    ds = Dataset(name="t", window_size=WINDOW_SIZE, scaler=scaler)
    ds.dataset = df_clean.copy()
    ds.corrupt(corruption_type=corrupt_type)
    w3d = make_windows(ds.dataset, WINDOW_SIZE)
    norm = scaler.transform(w3d.reshape(-1, INPUT_SIZE)).reshape(w3d.shape)
    nw = len(norm)
    lbl = np.zeros(nw, dtype=int)
    for col, idx_arr in ds.anomalies.items():
        for ri in idx_arr.astype(int):
            lbl[max(0, ri - WINDOW_SIZE + 1):min(nw, ri + 1)] = 1
    return norm, lbl


def window_scores(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(DEVICE)
            out, _ = model(x)
            scores.append(torch.mean(torch.abs(out - x), dim=(1, 2)).cpu().numpy())
    return np.concatenate(scores)


def best_threshold(scores, labels):
    try:
        _, _, ths = roc_curve(labels, scores)
        best_f1, best_t = 0, ths[0]
        for t in ths:
            f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return float(best_t), best_f1
    except Exception:
        return float(np.percentile(scores, 95)), 0.0


def get_metrics(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    return dict(
        auc=roc_auc_score(y_true, y_scores),
        f1=f1_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
    )


def train_es(model, tr_loader, va_loader, label):
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()
    best_v, wait, best_state = float("inf"), 0, None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        tl = 0
        for (x,) in tr_loader:
            x = x.to(DEVICE).float()
            opt.zero_grad()
            out, _ = model(x)
            loss = crit(out, x)
            loss.backward()
            opt.step()
            tl += loss.item()
        model.eval()
        vl = 0
        with torch.no_grad():
            for (x,) in va_loader:
                x = x.to(DEVICE).float()
                out, _ = model(x)
                vl += crit(out, x).item()
        avg_v = vl / len(va_loader)
        elapsed = (time.time() - t0) / 60
        print(f"  [{label}] epoch {ep+1:02d}/{EPOCHS}  train={tl/len(tr_loader):.5f}  val={avg_v:.5f}  ({elapsed:.1f} min)")
        if avg_v < best_v - 1e-6:
            best_v, wait = avg_v, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stop at epoch {ep+1}.")
                break
    if best_state:
        model.load_state_dict(best_state)
    return best_v


def mean_ci(values, confidence=0.95):
    values = np.array(values, dtype=float)
    n = len(values)
    if n < 2:
        return float(values[0]) if n == 1 else float("nan"), 0.0
    m = values.mean()
    sem = stats.sem(values)
    h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return float(m), float(h)


if __name__ == "__main__":
    print("Loading data from:", DATA_PATH)
    full_df = pd.read_csv(DATA_PATH, sep=";")
    n = len(full_df)
    tr_end = int(n * 0.70)
    va_end = int(n * 0.80)
    df_train = full_df.iloc[:tr_end].reset_index(drop=True)
    df_val = full_df.iloc[tr_end:va_end].reset_index(drop=True)
    df_test = full_df.iloc[va_end:].reset_index(drop=True)
    print(f"Split -> train:{len(df_train)} ({len(df_train)/n:.1%})  "
          f"val:{len(df_val)} ({len(df_val)/n:.1%})  "
          f"test:{len(df_test)} ({len(df_test)/n:.1%})")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_w = make_windows(df_train, WINDOW_SIZE)
    scaler.fit(train_w.reshape(-1, INPUT_SIZE))

    def norm(w):
        s = w.shape
        return scaler.transform(w.reshape(-1, INPUT_SIZE)).reshape(s)

    train_norm = norm(train_w)
    val_norm = norm(make_windows(df_val, WINDOW_SIZE))
    train_loader = make_loader(train_norm, shuffle=True)
    val_loader = make_loader(val_norm)
    print(f"Train windows: {len(train_norm)}  Val windows: {len(val_norm)}")

    # per-seed results: {model: {anomaly_type: {metric: [values across seeds]}}}
    # CHECKPOINT: resume from results_raw.json if it already has some seeds done.
    checkpoint_path = f"{RESULTS_DIR}/results_raw.json"
    completed_seeds_path = f"{RESULTS_DIR}/completed_seeds.json"
    if os.path.exists(checkpoint_path) and os.path.exists(completed_seeds_path):
        with open(checkpoint_path) as f:
            all_results = json.load(f)
        with open(completed_seeds_path) as f:
            completed_seeds = set(json.load(f))
        print(f"Resuming: found checkpoint with completed seeds {sorted(completed_seeds)}")
    else:
        all_results = {"AE-LSTM-ATT": {at: {"auc": [], "f1": [], "precision": [], "recall": []} for at in ANOMALY_TYPES},
                       "LSTM-AE": {at: {"auc": [], "f1": [], "precision": [], "recall": []} for at in ANOMALY_TYPES},
                       "Isolation Forest": {at: {"auc": [], "f1": [], "precision": [], "recall": []} for at in ANOMALY_TYPES}}
        completed_seeds = set()

    for seed in SEEDS:
        if seed in completed_seeds:
            print(f"\n===== SEED {seed} (already done, skipping) =====")
            continue
        print(f"\n===== SEED {seed} =====")
        try:
            set_seed(seed)

            att = AE_LSTM_ATT(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, WINDOW_SIZE, DROPOUT).to(DEVICE)
            train_es(att, train_loader, val_loader, "AE-LSTM-ATT")

            base = LSTMAEBaseline(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, WINDOW_SIZE, DROPOUT).to(DEVICE)
            train_es(base, train_loader, val_loader, "LSTM-AE")

            iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=seed, n_jobs=-1)
            iso.fit(train_norm.reshape(len(train_norm), -1))
            def iso_scores(data):
                return -iso.score_samples(data.reshape(len(data), -1))

            for at in ANOMALY_TYPES:
                val_anom_norm, val_lbl = make_anomaly_set(df_val.copy(), at, scaler)
                test_anom_norm, test_lbl = make_anomaly_set(df_test.copy(), at, scaler)

                for name, model_scores_val, model_scores_test in [
                    ("AE-LSTM-ATT",
                     np.concatenate([window_scores(att, val_loader), window_scores(att, make_loader(val_anom_norm))]),
                     window_scores(att, make_loader(test_anom_norm))),
                    ("LSTM-AE",
                     np.concatenate([window_scores(base, val_loader), window_scores(base, make_loader(val_anom_norm))]),
                     window_scores(base, make_loader(test_anom_norm))),
                    ("Isolation Forest",
                     np.concatenate([iso_scores(val_norm), iso_scores(val_anom_norm)]),
                     iso_scores(test_anom_norm)),
                ]:
                    val_labels = np.concatenate([np.zeros(len(val_norm)), val_lbl])
                    thr, _ = best_threshold(model_scores_val, val_labels)
                    m = get_metrics(test_lbl, model_scores_test, thr)
                    for k in ("auc", "f1", "precision", "recall"):
                        all_results[name][at][k].append(m[k])
                    print(f"  [{at}] {name:<18} auc={m['auc']:.3f} f1={m['f1']:.3f} p={m['precision']:.3f} r={m['recall']:.3f}")

            # CHECKPOINT: save after every seed finishes, so a crash/timeout only
            # costs the current seed, not the whole run.
            completed_seeds.add(seed)
            with open(checkpoint_path, "w") as f:
                json.dump(all_results, f, indent=2, default=float)
            with open(completed_seeds_path, "w") as f:
                json.dump(sorted(completed_seeds), f)
            print(f"  [checkpoint saved after seed {seed}: {sorted(completed_seeds)}]")

        except Exception as e:
            print(f"!!! SEED {seed} FAILED with error: {type(e).__name__}: {e}")
            print(f"!!! This seed's results are NOT saved. Re-running the script will")
            print(f"!!! resume from the last successful checkpoint ({sorted(completed_seeds)}) and retry this seed.")
            continue

    # ---- summarize: mean +/- 95% CI per model/type/metric ----
    summary_rows = []
    for model_name, types in all_results.items():
        for at, metrics in types.items():
            row = {"Model": model_name, "Anomaly": at, "n_seeds": len(metrics["auc"])}
            for k, vals in metrics.items():
                m, ci = mean_ci(vals)
                row[f"{k}_mean"] = round(m, 4)
                row[f"{k}_ci95"] = round(ci, 4)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{RESULTS_DIR}/results_meanCI.csv", index=False)
    with open(f"{RESULTS_DIR}/results_raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print("\n=== FINAL: mean +/- 95% CI across", len(SEEDS), "seeds ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {RESULTS_DIR}/results_meanCI.csv and {RESULTS_DIR}/results_raw.json")
