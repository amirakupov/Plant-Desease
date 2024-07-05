import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def load_data(data_dir, image_size=(96, 96)):
    healthy_categories = [
        "Pepper__bell___healthy",
        "Potato___healthy",
        "Tomato_healthy"
    ]
    diseased_categories = [
        "Pepper__bell___Bacterial_spot",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato__Target_Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato__Tomato_mosaic_virus"
    ]

    data = []
    labels = []

    for category in healthy_categories:
        path = os.path.join(data_dir, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array, image_size)
                data.append(img_array)
                labels.append(0)  # Healthy label
            except Exception as e:
                pass

    for category in diseased_categories:
        path = os.path.join(data_dir, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array, image_size)
                data.append(img_array)
                labels.append(1)  # Diseased label
            except Exception as e:
                pass

    return np.array(data), np.array(labels)


def prepare_datasets(data, labels, batch_size, buffer_size=10000):
    # Create a tf.data dataset from the arrays
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))

    # Shuffle, batch, and prefetch the dataset
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def get_data(data_dir, batch_size):
    X, y = load_data(data_dir)
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = prepare_datasets(X_train, y_train, batch_size)
    test_dataset = prepare_datasets(X_test, y_test, batch_size)

    return train_dataset, test_dataset, class_weights


if __name__ == "__main__":
    data_dir = '../data'
    batch_size = 32
    train_dataset, test_dataset, class_weights = get_data(data_dir, batch_size)
    for images, labels in train_dataset.take(1):
        print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    print(f"Class weights: {class_weights}")
