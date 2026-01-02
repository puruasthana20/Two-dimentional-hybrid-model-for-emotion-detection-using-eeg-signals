import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
import os


df = pd.read_pickle("data/dreamer_trials.pkl")

def extract_features(eeg_trial):
    features = []
    for ch in eeg_trial:
        features.append(np.mean(ch))
        features.append(np.std(ch))
        features.append(skew(ch))
        features.append(kurtosis(ch))
        features.append(entropy(np.abs(np.fft.fft(ch))))
    return np.array(features)


X = np.array([extract_features(eeg) for eeg in df['EEG']])
y_val = df['Valence'].values
y_arousal = df['Arousal'].values


def to_emotion(v, a):
    if v >= 3 and a >= 3: return "Happy"
    elif v < 3 and a >= 3: return "Angry"
    elif v < 3 and a < 3: return "Sad"
    else: return "Relaxed"

y_emotion = np.array([to_emotion(v, a) for v, a in zip(y_val, y_arousal)])


os.makedirs("data", exist_ok=True)
np.save("data/X_features.npy", X)
np.save("data/y_valence.npy", y_val)
np.save("data/y_emotion.npy", y_emotion)

print(" Features saved to data/X_features.npy")
print(" Emotion labels saved to data/y_emotion.npy")
