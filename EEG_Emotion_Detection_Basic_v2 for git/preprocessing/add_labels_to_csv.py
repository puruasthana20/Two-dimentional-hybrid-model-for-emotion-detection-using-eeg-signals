import pandas as pd
import numpy as np
import scipy.io


df = pd.read_csv('data/dreamer_eeg_data.csv')


mat = scipy.io.loadmat('data/DREAMER.mat')
dreamer = mat['DREAMER']


participants = dreamer['Data'][0, 0][0]
print(f"Total Participants: {len(participants)}")

valence = []
arousal = []


for participant in participants:
    val = participant['ScoreValence'][0][0]
    arous = participant['ScoreArousal'][0][0]

    valence.append(val)
    arousal.append(arous)

valence = np.array(valence)
arousal = np.array(arousal)


repeat_times = len(df) // len(valence)

valence_repeated = np.repeat(valence, repeat_times)
arousal_repeated = np.repeat(arousal, repeat_times)


valence_repeated = valence_repeated[:len(df)]
arousal_repeated = arousal_repeated[:len(df)]


df['Valence'] = valence_repeated
df['Arousal'] = arousal_repeated


df.to_csv('data/dreamer_eeg_data_labeled.csv', index=False)

print("✅ Labels added successfully!")
