
import pandas as pd
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=128.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def preprocess_eeg(file_path):
    df = pd.read_csv(file_path)
    df['alpha'] = bandpass_filter(df['alpha'])
    df['beta'] = bandpass_filter(df['beta'])
    df['theta'] = bandpass_filter(df['theta'])
    return df
