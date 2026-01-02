
from preprocessing.preprocess import preprocess_eeg
from features.extract_features import extract_features
from models.train_model import train_model

if __name__ == '__main__':
    train_model()
    print("Model trained. Run GUI using gui/app_gui.py")
