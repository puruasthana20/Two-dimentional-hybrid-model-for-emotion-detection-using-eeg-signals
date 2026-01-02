import mne

raw = mne.io.read_raw_edf("data/data_0003_raw.edf", preload=True, verbose=False)
eeg_data = raw.get_data()  # shape: (n_channels, n_samples)

print("Channels:", raw.ch_names)
print("Sampling rate:", raw.info['sfreq'])
print("Shape:", eeg_data.shape)  # e.g., (14, 8064)
