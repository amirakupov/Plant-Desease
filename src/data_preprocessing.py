import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir):
    categories = [category for category in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, category))]
    data = []
    labels = []

    for category in categories:
        class_num = categories.index(category)
        path = os.path.join(data_dir, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array, (128, 128))
                data.append(img_array)
                labels.append(class_num)
            except Exception as e:
                pass

    return np.array(data), np.array(labels)

def augment_data(X_train):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    datagen.fit(X_train)
    return datagen

def get_data(data_dir):
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    datagen = augment_data(X_train)
    return X_train, X_test, y_train, y_test, datagen

if __name__ == "__main__":
    data_dir = '/Users/amirakupov/Desktop/projects/plant_desease/data'  # Update this path to your dataset location
    X_train, X_test, y_train, y_test, datagen = get_data(data_dir)
    print(f"Data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")




