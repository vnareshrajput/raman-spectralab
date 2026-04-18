import streamlit as st
import matplotlib.pyplot as plt
from preprocessing import *

st.set_page_config(layout="wide")

st.title("🔬 Raman Spectra Processing Tool")

# -------------------------------
# SESSION STATE (STORE FILE)
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload .txt or .csv file")

if uploaded_file:
    st.session_state.df = load_file(uploaded_file)

# -------------------------------
# STOP if no file
# -------------------------------
if st.session_state.df is None:
    st.warning("Please upload a file")
    st.stop()

df = st.session_state.df

raman_shift = df.iloc[:, 0]
intensity = df.iloc[:, 1:]

# -------------------------------
# PARAMETERS (LIVE)
# -------------------------------
st.sidebar.header("⚙️ Parameters")

window_length = st.sidebar.slider("Window Length", 3, 21, 7, step=2)
poly_order = st.sidebar.slider("Polynomial Order", 1, 5, 2)

lam = st.sidebar.number_input("Lambda (λ)", value=1000.0)
p = st.sidebar.number_input("p", value=0.01)
niter = st.sidebar.slider("Iterations", 1, 20, 10)

roi_min = st.sidebar.number_input("ROI Min", value=600)
roi_max = st.sidebar.number_input("ROI Max", value=3100)

# -------------------------------
# PIPELINE (ALWAYS RUNS)
# -------------------------------

# 1️⃣ RAW
st.subheader("1️⃣ Raw Spectra")
fig, ax = plt.subplots()
ax.plot(raman_shift, intensity, alpha=0.5)
st.pyplot(fig)

# 2️⃣ ROI
raman_shift_roi, spectra_roi = apply_roi(df, roi_min, roi_max)

st.subheader("2️⃣ ROI Spectra")
fig, ax = plt.subplots()
ax.plot(raman_shift_roi, spectra_roi, alpha=0.5)
st.pyplot(fig)

# 3️⃣ SMOOTH
spectra_smooth = smooth_data(spectra_roi, window_length, poly_order)

st.subheader("3️⃣ Smoothed")
fig, ax = plt.subplots()
ax.plot(raman_shift_roi, spectra_smooth, alpha=0.7)
st.pyplot(fig)

# 4️⃣ BASELINE
spectra_corrected = baseline_correction(spectra_smooth, lam, p, niter)

st.subheader("4️⃣ Baseline Corrected")
fig, ax = plt.subplots()
ax.plot(raman_shift_roi, spectra_corrected, alpha=0.7)
st.pyplot(fig)

# 5️⃣ NORMALIZATION
normalized_df = normalize_data(spectra_corrected)

st.subheader("5️⃣ Normalized")
fig, ax = plt.subplots()
ax.plot(raman_shift_roi, normalized_df, alpha=0.7)
st.pyplot(fig)

# 6️⃣ MEAN
df_out, row_mean = calculate_mean(df.copy(), normalized_df)

st.subheader("6️⃣ Mean Spectrum")
fig, ax = plt.subplots()
ax.plot(raman_shift_roi, row_mean, linewidth=2)
st.pyplot(fig)

# -------------------------------
# DOWNLOAD
# -------------------------------
csv = df_out.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Processed CSV",
    csv,
    "processed_raman.csv",
    "text/csv"
)