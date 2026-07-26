#!/usr/bin/env python3
"""
Attention weight verification for AE-LSTM-ATT.

This tests the SPECIFIC empirical claim flagged in Section 4.3 of the
manuscript: "the highest attention weights consistently concentrated on
time steps immediately preceding fault onset."

It trains one AE-LSTM-ATT model, finds true-positive test windows (windows
correctly flagged as anomalous) for each anomaly type, extracts the real
attention weights the model produced for those windows, and checks whether
attention actually concentrates near the true fault-onset position within
each window -- rather than assuming it does.

REQUIRES: ae_lstm_att.py and Dataset.py (your real files) in the same
directory or on sys.path. Set INDRAD_DATA_PATH as in multiseed_eval.py.

Outputs:
  - Printed summary: for each anomaly type, the mean/median distance (in
    time steps) between each true-positive window's peak attention position
    and its actual fault-onset position, plus what fraction of TP windows
    have peak attention within +/-5 steps of onset.
  - attention_analysis.png: example attention heatmaps (one TP window per
    anomaly type) with the true onset position marked, plus a summary plot
    of the offset distribution across all TP windows.

HONEST NOTE: this either confirms or refutes the Section 4.3 claim. If the
offsets are large/random, the claim is not supported and should be removed
or rewritten to describe what is actually observed.
"""
import os, sys, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.getcwd())
from ae_lstm_att import AE_LSTM_ATT
from Dataset import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

DEFAULT_DATA_PATH = "dataset_train/train_dataset_nominal.csv"
DATA_PATH = os.environ.get("INDRAD_DATA_PATH", DEFAULT_DATA_PATH)

WINDOW_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT = 50, 64, 2, 0.3
WEIGHT_DECAY, EPOCHS, PATIENCE, BATCH_SIZE, LR = 1e-4, 30, 3, 256, 1e-3
SEED = 42
ANOMALY_TYPES = ["spike", "step", "freeze_zero", "freeze_last_value"]
INPUT_SIZE = 21


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_windows(df, w):
    d = df.values.astype(np.float32)
    idx = np.arange(w)[None, :] + np.arange(len(d) - w + 1)[:, None]
    return d[idx]


def make_label_windows(labels, w):
    idx = np.arange(w)[None, :] + np.arange(len(labels) - w + 1)[:, None]
    return labels[idx]  # [n_windows, w] -- per-timestep labels WITHIN each window


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
    raw_labels = np.zeros(len(ds.dataset), dtype=int)
    for col, idx_arr in ds.anomalies.items():
        raw_labels[idx_arr.astype(int)] = 1
    window_lbls = make_label_windows(raw_labels, WINDOW_SIZE)  # [nw, WINDOW_SIZE]
    binary_lbls = (window_lbls.sum(axis=1) > 0).astype(int)
    return norm, binary_lbls, window_lbls


def window_scores_and_attn(model, X):
    model.eval()
    scores, attns = [], []
    with torch.no_grad():
        for i in range(0, X.shape[0], 128):
            xb = torch.tensor(X[i:i+128], dtype=torch.float32).to(DEVICE)
            out, attn = model(xb)
            e = torch.mean(torch.abs(out - xb), dim=(1, 2)).cpu().numpy()
            scores.append(e)
            attns.append(attn.cpu().numpy())  # [B, T, T] decoder_step x encoder_step
    return np.concatenate(scores), np.concatenate(attns, axis=0)


def train_es(model, tr_loader, va_loader):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()
    best_v, wait, best_state = float("inf"), 0, None
    for ep in range(EPOCHS):
        model.train()
        for (x,) in tr_loader:
            x = x.to(DEVICE).float()
            opt.zero_grad()
            out, _ = model(x)
            loss = crit(out, x)
            loss.backward()
            opt.step()
        model.eval()
        vl = 0
        with torch.no_grad():
            for (x,) in va_loader:
                x = x.to(DEVICE).float()
                out, _ = model(x)
                vl += crit(out, x).item()
        avg_v = vl / len(va_loader)
        print(f"  epoch {ep+1:02d}/{EPOCHS}  val={avg_v:.5f}")
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
    return model


def best_threshold(scores, labels):
    _, _, ths = roc_curve(labels, scores)
    best_f1, best_t = 0, ths[0]
    for t in ths:
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


if __name__ == "__main__":
    print("Loading data from:", DATA_PATH)
    full_df = pd.read_csv(DATA_PATH, sep=";")
    n = len(full_df)
    tr_end, va_end = int(n * 0.70), int(n * 0.80)
    df_train = full_df.iloc[:tr_end].reset_index(drop=True)
    df_val = full_df.iloc[tr_end:va_end].reset_index(drop=True)
    df_test = full_df.iloc[va_end:].reset_index(drop=True)

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

    print("Training AE-LSTM-ATT (seed 42) for attention analysis...")
    set_seed(SEED)
    model = AE_LSTM_ATT(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, WINDOW_SIZE, DROPOUT).to(DEVICE)
    model = train_es(model, train_loader, val_loader)
    torch.save(model.state_dict(), "ae_lstm_att_seed42.pth")
    print("Saved ae_lstm_att_seed42.pth")

    val_scores, _ = window_scores_and_attn(model, val_norm)

    fig, axes = plt.subplots(2, len(ANOMALY_TYPES), figsize=(5*len(ANOMALY_TYPES), 8))
    summary = {}

    for i, at in enumerate(ANOMALY_TYPES):
        print(f"\n--- {at} ---")
        val_anom_norm, val_lbl, _ = make_anomaly_set(df_val.copy(), at, scaler)
        test_anom_norm, test_lbl, test_window_lbls = make_anomaly_set(df_test.copy(), at, scaler)

        val_anom_scores, _ = window_scores_and_attn(model, val_anom_norm)
        all_val_scores = np.concatenate([val_scores, val_anom_scores])
        all_val_labels = np.concatenate([np.zeros(len(val_scores)), val_lbl])
        thr = best_threshold(all_val_scores, all_val_labels)

        test_scores, test_attn = window_scores_and_attn(model, test_anom_norm)
        test_pred = (test_scores >= thr).astype(int)

        tp_idx = np.where((test_pred == 1) & (test_lbl == 1))[0]
        print(f"  True positives: {len(tp_idx)} / {len(test_lbl)} anomalous windows")

        offsets = []
        for idx in tp_idx:
            wlabels = test_window_lbls[idx]  # per-timestep labels within this window
            onset_candidates = np.where(wlabels == 1)[0]
            if len(onset_candidates) == 0:
                continue
            onset_pos = onset_candidates[0]  # first anomalous timestep within window

            attn_matrix = test_attn[idx]  # [T, T] decoder_step x encoder_step
            avg_attn_profile = attn_matrix.mean(axis=0)  # average over decoder steps -> [T]
            peak_pos = int(np.argmax(avg_attn_profile))

            offsets.append(peak_pos - onset_pos)

        offsets = np.array(offsets)
        if len(offsets) > 0:
            mean_abs_offset = np.mean(np.abs(offsets))
            median_offset = np.median(offsets)
            frac_within_5 = np.mean(np.abs(offsets) <= 5)
            print(f"  Mean |offset| (peak attn vs onset): {mean_abs_offset:.1f} steps")
            print(f"  Median offset: {median_offset:.1f} steps")
            print(f"  Fraction of TP windows with peak attn within +/-5 steps of onset: {frac_within_5:.1%}")
            summary[at] = dict(n_tp=len(tp_idx), n_scored=len(offsets),
                                mean_abs_offset=mean_abs_offset, median_offset=median_offset,
                                frac_within_5=frac_within_5)
        else:
            print("  No true positives with a clear onset found for this type.")
            summary[at] = dict(n_tp=len(tp_idx), n_scored=0)

        # Example heatmap: first TP window for this anomaly type
        ax_top = axes[0, i]
        if len(tp_idx) > 0:
            example = test_attn[tp_idx[0]]
            im = ax_top.imshow(example, aspect='auto', cmap='viridis')
            wlabels = test_window_lbls[tp_idx[0]]
            onset_candidates = np.where(wlabels == 1)[0]
            if len(onset_candidates) > 0:
                ax_top.axvline(onset_candidates[0], color='red', linestyle='--', label='true onset')
                ax_top.legend(fontsize=7)
        ax_top.set_title(f"{at}\n(example TP window)", fontsize=10)
        ax_top.set_xlabel("Encoder time step")
        ax_top.set_ylabel("Decoder time step")

        ax_bot = axes[1, i]
        if len(offsets) > 0:
            ax_bot.hist(offsets, bins=20, color='steelblue', alpha=0.8)
            ax_bot.axvline(0, color='red', linestyle='--', label='onset (offset=0)')
            ax_bot.legend(fontsize=7)
        ax_bot.set_xlabel("Peak attention position - onset position")
        ax_bot.set_ylabel("Count (TP windows)")

    plt.tight_layout()
    plt.savefig("attention_analysis.png", dpi=150, bbox_inches='tight')
    print("\nSaved attention_analysis.png")

    print("\n=== SUMMARY (send this back for the Section 4.3 rewrite) ===")
    for at, s in summary.items():
        print(at, s)
