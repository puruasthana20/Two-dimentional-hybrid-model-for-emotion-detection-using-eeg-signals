

import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


model = joblib.load("data/emotion_model.pkl")


X = np.load("data/X_features.npy", allow_pickle=True)
y_true = np.load("data/y_emotion.npy", allow_pickle=True)
y_pred = model.predict(X)


labels = ["Happy", "Angry", "Relaxed", "Sad"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Emotion Classification")
plt.show()
