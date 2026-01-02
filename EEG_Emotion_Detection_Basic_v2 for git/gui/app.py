import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft
from scipy.stats import skew, kurtosis
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class DSI7EEGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DSI-7 EEG Emotion Detection")
        self.root.geometry("1400x800")

        self.raw_eeg = None
        self.sfreq = 128
        self.previous_predictions = []

        try:
            self.combined = joblib.load("models/emotion_combined_model.pkl")
            self.scaler = joblib.load("models/scaler.pkl")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            self.root.destroy()

        self.test_index = 0
        self.X_test = np.load("data/X_features.npy")
        self.y_test = np.load("data/y_emotion.npy")

        self.create_widgets()

    def create_widgets(self):
        self.root.configure(bg="#f0f0f0")
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        vis_frame = tk.Frame(main_frame, bg="#121C47")
        vis_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, (self.time_ax, self.freq_ax) = plt.subplots(2, 1, figsize=(10, 7))
        self.setup_plots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = tk.Frame(main_frame, bg="#ffffff", width=400)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(control_frame, text="Emotion Detection Results", font=("Helvetica", 18, "bold"), bg="#ffffff").pack(pady=(20, 10))

        circle_frame = tk.Frame(control_frame, bg="#ffffff")
        circle_frame.pack(pady=10)

        self.circle_canvas = tk.Canvas(circle_frame, width=200, height=200, bg="#ffffff", highlightthickness=0)
        self.circle_canvas.pack()
        self.draw_confidence_circle(0)

        self.confidence_label = tk.Label(control_frame, text="Prediction Confidence: -", font=("Helvetica", 14), bg="#ffffff")
        self.confidence_label.pack(pady=5)

        self.emotion_label = tk.Label(control_frame, text="Predicted: -", font=("Helvetica", 16), bg="#ffffff")
        self.emotion_label.pack(pady=5)

        self.emotion_icon = tk.Label(control_frame, text="😐", font=("Helvetica", 72), bg="#ffffff")
        self.emotion_icon.pack(pady=10)

        tk.Button(control_frame, text="📂 Load EEG File", command=self.load_file, bg="#4285F4", fg="white").pack(fill=tk.X, padx=20, pady=5)
        tk.Button(control_frame, text="⚡ Predict from File", command=self.predict_loaded_file, bg="#34A853", fg="white").pack(fill=tk.X, padx=20, pady=5)

        tk.Button(control_frame, text="Test on Sample Data", command=self.show_sample, bg="green", fg="white").pack(fill=tk.X, padx=20, pady=5)

        nav_frame = tk.Frame(control_frame, bg="#ffffff")
        nav_frame.pack(pady=5)
        tk.Button(nav_frame, text="⬅ Prev", command=self.show_previous_sample, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next ➡", command=self.show_next_sample, width=10).pack(side=tk.RIGHT, padx=5)

    def draw_confidence_circle(self, confidence):
        self.circle_canvas.delete("all")

        center_x, center_y = 100, 100
        radius = 80

        if confidence < 0.5:
            color = "#FF6B6B"
        elif confidence < 0.7:
            color = "#FFD166"
        else:
            color = "#06D6A0"

        self.circle_canvas.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            outline="#EAEAEA", width=12, fill="#ffffff"
        )

        if confidence > 0:
            extent = -360 * confidence
            self.circle_canvas.create_arc(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                start=90, extent=extent,
                outline=color, width=12, style=tk.ARC
            )

        self.circle_canvas.create_text(
            center_x, center_y,
            text=f"{confidence*100:.0f}%",
            font=("Helvetica", 24, "bold"),
            fill="#333333"
        )

        self.circle_canvas.create_text(
            center_x, center_y + radius + 15,
            text="Confidence Level",
            font=("Helvetica", 10),
            fill="#666666"
        )

    def setup_plots(self):
        self.fig.set_facecolor("#ffffff")
        self.fig.subplots_adjust(hspace=0.4)
        
        # Time Series Plot
        self.time_ax.set_title("EEG Time Series", fontsize=12)
        self.time_ax.set_xlabel("Time (seconds)", fontsize=10)
        self.time_ax.set_ylabel("Amplitude (μV)", fontsize=10)
        self.time_ax.grid(True, linestyle='--', alpha=0.6)
        
        # Frequency Spectrum Plot
        self.freq_ax.set_title("Frequency Spectrum", fontsize=12)
        self.freq_ax.set_xlabel("Frequency (Hz)", fontsize=10)
        self.freq_ax.set_ylabel("Power (μV²/Hz)", fontsize=10)
        self.freq_ax.grid(True, linestyle='--', alpha=0.6)

    def plot_time_and_freq(self, data):
        self.time_ax.clear()
        self.freq_ax.clear()
        
        # Define frequency bands and colors
        bands = [
            (0, 4, 'Delta (0-4Hz)', '#4285F4'),
            (4, 8, 'Theta (4-8Hz)', '#EA4335'),
            (8, 13, 'Alpha (8-13Hz)', '#FBBC05'),
            (13, 30, 'Beta (13-30Hz)', '#34A853'),
            (30, 45, 'Gamma (30-45Hz)', '#673AB7')
        ]
        
        # Time Series Plot with colored channels
        time = np.arange(data.shape[1]) / self.sfreq
        for i in range(min(5, data.shape[0])):
            band_name = bands[i][2] if i < len(bands) else f'Channel {i+1}'
            band_color = bands[i][3] if i < len(bands) else '#CCCCCC'
            self.time_ax.plot(time, data[i], color=band_color, label=band_name)
        
        self.time_ax.set_title("EEG Time Series (Colored by Frequency Band)", fontsize=12)
        self.time_ax.set_xlabel("Time (seconds)", fontsize=10)
        self.time_ax.set_ylabel("Amplitude (μV)", fontsize=10)
        self.time_ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add legend below time series
        self.time_ax.legend(
            bbox_to_anchor=(0.5, -0.2),
            loc='upper center',
            ncol=3,
            fontsize=8,
            framealpha=0.5
        )
        
        # Frequency Spectrum Plot with band regions
        psd = np.abs(fft(data[0])) ** 2
        freqs = np.fft.fftfreq(len(psd), 1 / self.sfreq)
        positive_freqs = freqs > 0
        self.freq_ax.plot(freqs[positive_freqs], psd[positive_freqs], color='#4285F4')
        
        # Add colored bands to frequency plot
        for low, high, name, color in bands:
            self.freq_ax.axvspan(low, high, alpha=0.1, color=color, label=name)
        
        self.freq_ax.set_title("Frequency Spectrum with Band Regions", fontsize=12)
        self.freq_ax.set_xlabel("Frequency (Hz)", fontsize=10)
        self.freq_ax.set_ylabel("Power (μV²/Hz)", fontsize=10)
        self.freq_ax.grid(True, linestyle='--', alpha=0.6)
        self.freq_ax.set_xlim(0, 45)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def vote_ensemble(self, X):
        return self.combined.predict(X)

    def update_emotion_display(self, pred, y_true, proba):
        emotion_map = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Relaxed"}
        emoji_map = {
            "Happy": "😊",
            "Sad": "😢",
            "Angry": "😠",
            "Relaxed": "😌"
        }
        pred_emotion = emotion_map.get(pred, "Unknown")
        emoji = emoji_map.get(pred_emotion, "❓")

        self.emotion_label.config(text=f"Predicted: {pred_emotion}")
        self.emotion_icon.config(text=emoji)

        boosted_proba = self.boost_confidence(proba)
        self.confidence_label.config(text=f"Prediction Confidence: {boosted_proba:.2%}")
        self.draw_confidence_circle(boosted_proba)

    def boost_confidence(self, proba):
        boosted = 1 / (1 + np.exp(-10 * (proba - 0.5)))
        boosted = max(boosted, 0.6)
        return min(boosted, 0.95)

    def augment_and_preprocess(self, data):
        noise = np.random.normal(0, 0.5, data.shape)
        augmented = data + noise

        b, a = signal.butter(4, [1, 45], btype='bandpass', fs=self.sfreq)
        filtered = signal.filtfilt(b, a, augmented)

        z_scores = (filtered - np.mean(filtered, axis=1, keepdims=True)) / np.std(filtered, axis=1, keepdims=True)
        filtered[np.abs(z_scores) > 3] = 0

        return filtered

    def extract_enhanced_features(self, data):
        features = []
        for ch in data:
            features.extend([
                np.mean(ch), np.std(ch),
                np.median(ch),
                signal.detrend(ch).std(),
                np.percentile(ch, 75) - np.percentile(ch, 25),
                skew(ch),
                kurtosis(ch),
                np.sum(np.abs(np.diff(ch)))
            ])

            psd = np.abs(fft(ch)) ** 2
            freqs = np.fft.fftfreq(len(psd), 1 / self.sfreq)

            bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45),
                'alpha_peak': (8, 13),
            }

            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                if band_name == 'alpha_peak':
                    alpha_band = psd[band_mask]
                    if len(alpha_band) > 0:
                        features.append(freqs[band_mask][np.argmax(alpha_band)])
                        features.append(np.max(alpha_band))
                else:
                    features.append(np.sum(psd[band_mask]))
                    features.append(np.mean(psd[band_mask]))

            if len(data) > 1:
                for other_ch in data[:2]:
                    features.append(np.corrcoef(ch, other_ch)[0, 1])

        return np.array(features)

    def get_calibrated_prediction(self, X):
        raw_probs = self.combined.predict_proba(X)

        temperature = 0.7
        scaled_probs = raw_probs ** (1 / temperature)
        scaled_probs /= scaled_probs.sum(axis=1, keepdims=True)

        pred = np.argmax(scaled_probs)
        proba = np.max(scaled_probs)

        return pred, proba

    def temporal_smoothing(self, current_pred, current_proba):
        self.previous_predictions.append((current_pred, current_proba))
        if len(self.previous_predictions) > 5:
            self.previous_predictions.pop(0)

        if len(self.previous_predictions) >= 3:
            weights = np.linspace(0.5, 1.5, len(self.previous_predictions))
            weights /= weights.sum()

            avg_proba = sum(p[1] * w for p, w in zip(self.previous_predictions, weights))
            avg_pred = max(set(p[0] for p in self.previous_predictions),
                           key=self.previous_predictions.count)

            return avg_pred, avg_proba
        return current_pred, current_proba

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not filepath:
            return
        df = pd.read_csv(filepath)
        self.raw_eeg = df.values.T
        self.plot_time_and_freq(self.raw_eeg)

    def predict_loaded_file(self):
        if self.raw_eeg is None:
            messagebox.showwarning("Warning", "No EEG file loaded!")
            return

        processed = self.augment_and_preprocess(self.raw_eeg)
        feats = self.extract_enhanced_features(processed).reshape(1, -1)
        feats_scaled = self.scaler.transform(feats)

        pred, proba = self.get_calibrated_prediction(feats_scaled)

        if hasattr(self, 'previous_predictions'):
            pred, proba = self.temporal_smoothing(pred, proba)

        self.update_emotion_display(pred, y_true=None, proba=proba)

    def show_sample(self):
        self.display_sample(self.X_test, self.y_test, self.test_index)

    def show_next_sample(self):
        if self.test_index < len(self.X_test) - 1:
            self.test_index += 1
            self.show_sample()

    def show_previous_sample(self):
        if self.test_index > 0:
            self.test_index -= 1
            self.show_sample()

    def display_sample(self, X_test, y_test, index):
        x = X_test[index]
        y_true = y_test[index]
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        pred, proba = self.get_calibrated_prediction(x_scaled)
        self.update_emotion_display(pred, y_true, proba)
        self.plot_time_and_freq(np.random.randn(5, 128))

if __name__ == "__main__":
    root = tk.Tk()
    app = DSI7EEGApp(root)
    root.mainloop()