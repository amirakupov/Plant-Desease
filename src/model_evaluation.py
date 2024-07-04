import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from data_preprocessing import get_data

def evaluate_model(model_path, data_dir):
    _, X_test, _, y_test, _ = get_data(data_dir)
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    data_dir = '../data'
    evaluate_model('plant_disease_model.h5', data_dir)
