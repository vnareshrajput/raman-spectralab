import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler
def load_file(file):
    if file.name.endswith(".txt"):

        # Read lines
        lines = file.readlines()
        lines = [line.decode("utf-8") for line in lines]

        # Find start
        start_index = 0
        for i, line in enumerate(lines):
            if "Begin Spectral Data" in line:
                start_index = i + 1
                break

        # 🔥 IMPORTANT FIX
        file.seek(0)

        df = pd.read_csv(file, sep=r"\s+", skiprows=start_index, header=None)

        # Handle variable columns
        if df.shape[1] == 2:
            df.columns = ["Raman_Shift", "Intensity"]
        else:
            cols = ["Raman_Shift"] + [f"Intensity_{i}" for i in range(1, df.shape[1])]
            df.columns = cols

    else:
        df = pd.read_csv(file)

    return df

def apply_roi(df, roi_min, roi_max):
    raman_shift = df.iloc[:, 0]
    intensity = df.iloc[:, 1:]

    mask = (raman_shift >= roi_min) & (raman_shift <= roi_max)

    return raman_shift[mask], intensity.loc[mask, :]


def smooth_data(data, window_length, poly_order):
    return data.apply(
        lambda col: savgol_filter(col, window_length, poly_order),
        axis=0
    )


def asls_baseline(y, lam, p, niter):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
    w = np.ones(L)

    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * (D.T @ D)
        z = spsolve(Z, W @ y)
        w = p * (y > z) + (1 - p) * (y <= z)

    return y - z


def baseline_correction(data, lam, p, niter):
    return data.apply(
        lambda col: asls_baseline(col.values, lam, p, niter),
        axis=0
    )


def normalize_data(data):
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(data)
    return pd.DataFrame(norm, columns=data.columns)


def calculate_mean(df, normalized_df):
    row_mean = normalized_df.mean(axis=1)
    df["Mean_Intensity"] = row_mean
    return df, row_mean