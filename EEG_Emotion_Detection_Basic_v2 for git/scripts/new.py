import tkinter as tk
from tkinter import filedialog
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft
from scipy.io import loadmat
import os
import threading
import time
import mne

# Load model and scaler
model = joblib.load("models/emotion_combined_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Feature extraction function
def extract_features(eeg_trial):
    features = []
    for ch in eeg_trial:
        features.append(np.mean(ch))
        features.append(np.std(ch))
        features.append(skew(ch))
        features.append(kurtosis(ch))
        features.append(entropy(np.abs(np.fft.fft(ch))))
    return np.array(features)

# Preprocessing (optional placeholder)
def preprocess_raw_eeg(data):
    return data  # Insert filtering or cleaning if needed

# Main GUI setup
root = tk.Tk()
root.title("Real-Time EEG Emotion Dashboard")
root.configure(bg="#04092D")
root.geometry("1300x750")

# Title
title = tk.Label(root, text="🧠 REAL-TIME EEG EMOTION DETECTION DASHBOARD", font=("Helvetica", 22), bg="#080725", fg="white")
title.pack(pady=15)

# EEG Frequency Plot
frame_plot = tk.Frame(root, bg="#121C47")
frame_plot.pack(side=tk.LEFT, padx=20, pady=10)
fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=100)
fig.patch.set_facecolor("#160B59")
ax.set_facecolor("#000000")
ax.set_title("EEG Frequency Spectrum", color="white")
ax.set_xlabel("Frequency (Hz)", color="white")
ax.set_ylabel("Amplitude", color="white")
ax.tick_params(colors="white")
canvas = FigureCanvasTkAgg(fig, master=frame_plot)
canvas.draw()
canvas.get_tk_widget().pack()

# Info panel
frame_info = tk.Frame(root, bg="#111111", width=400, height=600)
frame_info.pack(side=tk.RIGHT, padx=20, pady=20)
frame_info.pack_propagate(0)

emotion_label = tk.Label(frame_info, text="Predicted Emotion:", font=("Helvetica", 20, "bold"), bg="#111111", fg="white")
emotion_label.pack(pady=15)

emotion_output = tk.Label(frame_info, text="🙂", font=("Helvetica", 36, "bold"), bg="#111111", fg="yellow")
emotion_output.pack(pady=20)

valence_arousal = tk.Label(frame_info, text="File: Waiting...\nActual: --", font=("Helvetica", 16), bg="#111111", fg="white")
valence_arousal.pack(pady=15)

metrics_label = tk.Label(frame_info, text="Status: Monitoring folder for EEG files...", font=("Helvetica", 15), bg="#111111", fg="lightgreen")
metrics_label.pack(pady=20)

# EEG processing function
def process_eeg_file(filepath):
    try:
        ext = filepath.split('.')[-1].lower()
        if ext == 'csv':
            eeg_data = np.loadtxt(filepath, delimiter=",").T
        elif ext == 'edf':
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            eeg_data = raw.get_data()[:14]  # First 14 channels
        elif ext == 'mat':
            mat = loadmat(filepath)
            eeg_data = mat.get('EEG') or mat.get('data') or next(iter(mat.values()))
            if isinstance(eeg_data, dict):
                eeg_data = eeg_data['data']
        else:
            raise ValueError("Unsupported file format")

        eeg_data = preprocess_raw_eeg(eeg_data)
        eeg_data = eeg_data[:, :1280] if eeg_data.shape[1] > 1280 else eeg_data
        features = extract_features(eeg_data)
        scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(scaled)[0]

        # Update GUI
        emotion_output.config(text=f"🙂 {prediction}")
        valence_arousal.config(text=f"File: {os.path.basename(filepath)}\nActual: Unknown")
        metrics_label.config(text="Prediction Successful!", fg="lightgreen")

        # Update frequency plot
        ax.clear()
        signal = eeg_data[0][:512]
        freqs = np.fft.rfftfreq(len(signal), d=1/128)
        spectrum = np.abs(fft(signal))[:len(freqs)]
        ax.plot(freqs, spectrum, color='cyan')
        ax.set_title("EEG Channel 1 Frequency Spectrum", color="white")
        ax.set_xlabel("Frequency (Hz)", color="white")
        ax.set_ylabel("Amplitude", color="white")
        ax.set_facecolor("#000000")
        ax.tick_params(colors="white")
        fig.patch.set_facecolor("#160B59")
        canvas.draw()

    except Exception as e:
        emotion_output.config(text="❌ Error")
        valence_arousal.config(text=str(e))
        metrics_label.config(text="Prediction failed.", fg="red")

# Folder monitoring thread
def monitor_folder(folder="live_data"):
    seen = set()
    while True:
        try:
            files = [f for f in os.listdir(folder) if f.endswith((".csv", ".edf", ".mat"))]
            for fname in files:
                full_path = os.path.join(folder, fname)
                if full_path not in seen:
                    seen.add(full_path)
                    process_eeg_file(full_path)
        except Exception as e:
            metrics_label.config(text=f"Monitor error: {e}", fg="orange")
        time.sleep(5)

# Launch folder monitor
threading.Thread(target=monitor_folder, daemon=True).start()

# Start GUI
root.mainloop()
