import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


X = np.load("data/X_features.npy", allow_pickle=True)
y = np.load("data/y_emotion.npy", allow_pickle=True)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)
print(" Balanced Classes:", Counter(y_bal))

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

models = {
    " Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    " KNN": KNeighborsClassifier(n_neighbors=5),
    " Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    " SVM": SVC(kernel='rbf', class_weight='balanced'),
    " MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, early_stopping=True)
}


for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

   
    if name == " Random Forest":
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_bal))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_bal), yticklabels=np.unique(y_bal))
        plt.title("Confusion Matrix - Random Forest")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()
