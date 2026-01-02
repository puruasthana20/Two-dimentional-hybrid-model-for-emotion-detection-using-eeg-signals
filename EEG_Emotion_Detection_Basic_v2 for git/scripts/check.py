import scipy.io
import numpy as np
import pandas as pd

mat = scipy.io.loadmat("data/DREAMER.mat", struct_as_record=False, squeeze_me=True)
dreamer = mat['DREAMER']
subjects = dreamer.Data

all_trials = []

for i, subject in enumerate(subjects):
    eeg_trials = subject.EEG.stimuli
    valence = subject.ScoreValence
    arousal = subject.ScoreArousal
    gender = subject.Gender
    age = subject.Age

    for j, trial in enumerate(eeg_trials):
        eeg_array = np.array(trial).T  # shape: (14, variable)
        # Optional: minimum length check
        if eeg_array.shape[1] < 8000:
            continue
        all_trials.append({
            'Subject_ID': i + 1,
            'Gender': gender,
            'Age': age,
            'EEG': eeg_array,
            'Valence': float(valence[j]),
            'Arousal': float(arousal[j])
        })

df = pd.DataFrame(all_trials)

print(" Data shape:", df.shape)
if not df.empty:
    print(" First EEG shape:", df['EEG'].iloc[0].shape)
else:
    print(" No trials passed the filtering.")
