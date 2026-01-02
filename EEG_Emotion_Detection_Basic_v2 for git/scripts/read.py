import pandas as pd


df = pd.read_pickle("data/dreamer_trials.pkl")



pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', None)
print(df.head(100))





trial_0 = df.loc[0, 'EEG']
print("Trial 0 EEG shape:", trial_0.shape)  




