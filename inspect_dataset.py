import numpy as np
import pandas as pd
from data_loader import load_subject_raw_epochs

CHANNELS_22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz"
]

X_train, y_train, X_test, y_test = load_subject_raw_epochs(subject_id=1, fmin=4.0, fmax=38.0)

# 1. Full Manifest of ALL 576 Trials (288 Train + 288 Test)
rows = []
for idx in range(len(X_train)):
    rows.append({
        "trial_id": f"TRAIN_{idx:03d}",
        "split": "TRAIN",
        "session_tag": "0train (Day 1)",
        "movement_label": y_train[idx],
        "n_channels": X_train.shape[1],
        "n_timepoints": X_train.shape[2],
        "C3_std_uV": round(float(np.std(X_train[idx, 7]) * 1e6), 3),
        "C4_std_uV": round(float(np.std(X_train[idx, 11]) * 1e6), 3),
    })

for idx in range(len(X_test)):
    rows.append({
        "trial_id": f"TEST_{idx:03d}",
        "split": "TEST",
        "session_tag": "1test (Day 2)",
        "movement_label": y_test[idx],
        "n_channels": X_test.shape[1],
        "n_timepoints": X_test.shape[2],
        "C3_std_uV": round(float(np.std(X_test[idx, 7]) * 1e6), 3),
        "C4_std_uV": round(float(np.std(X_test[idx, 11]) * 1e6), 3),
    })

df_manifest = pd.DataFrame(rows)
df_manifest.to_csv("subject1_all_576_trials_manifest.csv", index=False)

# 2. All 1,001 time steps (full 4.0 seconds) of Train Trial 0
df_full_trial = pd.DataFrame(X_train[0].T, columns=CHANNELS_22)
df_full_trial.insert(0, "time_step", np.arange(X_train.shape[2]))
df_full_trial.insert(1, "time_sec", np.round(np.arange(X_train.shape[2]) / 250.0, 4))
df_full_trial.to_csv("subject1_trial0_full_4sec.csv", index=False)

print("Exported 2 files to your VS Code folder:")
print(f"1. subject1_all_576_trials_manifest.csv -> {len(df_manifest)} total rows (288 TRAIN + 288 TEST)")
print(f"2. subject1_trial0_full_4sec.csv -> {len(df_full_trial)} time steps x 22 EEG channels (Label: {y_train[0]})")