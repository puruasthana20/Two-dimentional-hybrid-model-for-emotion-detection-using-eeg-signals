
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter


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
print(" Feature matrix shape:", X.shape)


y_val = df['Valence'].values
X_train, X_test, y_train, y_test = train_test_split(X, y_val, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

print("\n Valence Regression Results:")
print(" R² Score:", r2_score(y_test, y_pred))
print(" MSE:", mean_squared_error(y_test, y_pred))




def to_emotion(val, arousal):
    if val >= 3 and arousal >= 3:
        return "Happy"
    elif val < 3 and arousal >= 3:
        return "Angry"
    elif val < 3 and arousal < 3:
        return "Sad"
    else:
        return "Relaxed"

df['Emotion'] = [to_emotion(v, a) for v, a in zip(df['Valence'], df['Arousal'])]
y_cls = df['Emotion'].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_scaled, y_cls)

print(" Balanced Emotion Counts:", Counter(y_bal))


X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_cls, y_train_cls)
y_pred_cls = clf.predict(X_test_cls)

print("\n Emotion Classification Results:")
print(" Classification Report:")
print(classification_report(y_test_cls, y_pred_cls))
