import numpy as np
import joblib
from collections import Counter
from sklearn.metrics import classification_report


model = joblib.load("models/emotion_combined_model.pkl")
scaler = joblib.load("models/scaler.pkl")
X = np.load("data/X_features.npy", allow_pickle=True)
y = np.load("data/y_emotion.npy", allow_pickle=True)

print(" Model, scaler, and data loaded successfully!")


sample_index = int(input("Enter sample index to predict (0 to {}): ".format(len(X) - 1)))


sample = scaler.transform(X[sample_index].reshape(1, -1))
pred = model.predict(sample)

print(f"\n Single Sample Prediction (index {sample_index}): {pred[0]}")
print(f" Actual label: {y[sample_index]}")


print("\n First 50 Sample Predictions:")
for i in range(300):
    sample_scaled = scaler.transform(X[i].reshape(1, -1))
    pred = model.predict(sample_scaled)
    print(f"Sample {i:02d} → Predicted: {pred[0]} | Actual: {y[i]}")


X_scaled = scaler.transform(X)
y_pred_all = model.predict(X_scaled)

print("\n Prediction distribution on full data:")
print(Counter(y_pred_all))