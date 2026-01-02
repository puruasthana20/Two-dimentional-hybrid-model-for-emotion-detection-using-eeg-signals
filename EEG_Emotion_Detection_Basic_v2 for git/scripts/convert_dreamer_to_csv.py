import scipy.io
import numpy as np
import csv


mat_data = scipy.io.loadmat('data/DREAMER.mat')
dreamer_data = mat_data['DREAMER']
subjects = dreamer_data[0, 0]['Data']


output_path = 'data/dreamer_eeg_data.csv'
header = ['Subject_ID', 'Gender', 'EEG_Data_Shape', 'Valence', 'Arousal']

with open(output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    for subj in subjects:
        subject_id = subj[0].item()
        gender = subj[1].item()
        eeg_data = subj[2]
        eeg_shape = eeg_data.shape
        
        
        valence = subj[3].item()
        arousal = subj[5].item()
        
        writer.writerow([subject_id, gender, str(eeg_shape), valence, arousal])

print(f"✅ CSV file written to {output_path}")