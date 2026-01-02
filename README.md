# EEG-Based Emotion Detection System 🧠📊

## 📌 Project Overview

This project focuses on **emotion recognition using EEG (Electroencephalogram) signals** by applying machine learning techniques on publicly available EEG datasets. The system predicts human emotional states based on **Valence–Arousal** dimensions and provides **confidence-based visualization** through a graphical interface.

The goal of this project is to demonstrate how raw EEG signals can be transformed into meaningful emotional insights using signal processing, feature extraction, ensemble learning, and an interactive GUI.

---

## 🎯 Problem Statement

Human emotions are complex and subjective. Traditional emotion recognition methods (facial expressions, speech) can be unreliable. EEG-based emotion detection offers a **direct neural perspective**, enabling more robust and real-time emotional state classification.

This project aims to:

* Analyze EEG signals for emotional patterns
* Classify emotions using machine learning models
* Provide interpretable confidence scores
* Visualize predictions through an intuitive GUI

---

## 🧠 Emotional Model Used

* **Valence**: Positive ↔ Negative emotions
* **Arousal**: Calm ↔ Excited state

> *Dominance was intentionally excluded to simplify classification and improve model stability.*

---

## 📂 Datasets Used

### 1. DREAMER Dataset

* EEG signals recorded while subjects watched emotion-eliciting videos
* Pre-labeled with Valence, Arousal, and Dominance scores
* High-quality multichannel EEG recordings

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Signal normalization
* Noise reduction
* Channel-wise feature extraction

### 2. Feature Extraction

* Statistical features (mean, variance, skewness)
* Frequency-domain features (band power)
* Time-domain EEG characteristics

### 3. Machine Learning Models

The system uses an **ensemble-based approach** combining:

* **K-Nearest Neighbors (KNN)** – captures local EEG patterns
* **Random Forest (RF)** – handles non-linearity and feature importance
* **Multi-Layer Perceptron (MLP)** – learns complex neural representations

### 4. Ensemble Strategy

* Majority voting / confidence-weighted prediction
* Improves robustness and generalization

---

## 🖥️ Application Interface

* Built using **Tkinter (Python GUI framework)**
* Displays:

  * Predicted emotion (Valence–Arousal)
  * Model confidence scores
  * EEG signal visualization

---

## 🛠️ Tech Stack

* **Programming Language**: Python
* **Libraries & Tools**:

  * NumPy, Pandas
  * Scikit-learn
  * Matplotlib
  * Tkinter
* **ML Techniques**: Ensemble Learning, Neural Networks

---

## 📊 Results

* Achieved reliable emotion classification accuracy on unseen EEG samples
* Ensemble model outperformed individual classifiers
* GUI improved interpretability and usability

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 🔮 Future Improvements

* Real-time EEG data integration
* Deep learning models (CNN / LSTM)
* Web-based interface (Streamlit / Flask)
* Emotion feedback-based adaptive systems

---

## 👨‍💻 Author

**Puru Asthana**
B.Tech Undergraduate
Jaypee University of Engineering and Technology

---

## 📜 License

This project is for academic and research purposes.

---

⭐ *If you find this project interesting, feel free to star the repository and explore further!*
