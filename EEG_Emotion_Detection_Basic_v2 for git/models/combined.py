import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib


X = np.load("data/X_features.npy", allow_pickle=True)  # shape: (n_samples, 70)
y = np.load("data/y_emotion.npy", allow_pickle=True)   # labels: ['Happy', 'Angry', 'Sad', 'Relaxed']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)
print("\n Balanced label counts:", Counter(y_bal))


X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)


rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)


voting_clf = VotingClassifier(estimators=[
    ('rf', rf),
    ('knn', knn),
    ('mlp', mlp)
], voting='soft')


voting_clf.fit(X_train, y_train)


y_pred = voting_clf.predict(X_test)
print("\n Combined Classifier Report:")
print(classification_report(y_test, y_pred))


joblib.dump(voting_clf, "models/emotion_combined_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\n Combined model and scaler saved successfully!")
