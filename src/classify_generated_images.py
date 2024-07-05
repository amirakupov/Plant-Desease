import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
import matplotlib.pyplot as plt


def classify_images(model_path, image_path, image_size=(128, 128)):
    model = load_model(model_path)

    # Load and preprocess the generated image
    generated_img = load_img(image_path)
    img_array = img_to_array(generated_img)

    # Resize image to match the model input shape
    img_array = cv2.resize(img_array, image_size)

    # Expand dimensions and normalize
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    prediction_prob = prediction[0][predicted_class]

    if predicted_class == 0:
        label = 'Healthy'
    else:
        label = 'Diseased'

    print(f'Prediction: {prediction}, Predicted Class: {predicted_class}, Probability: {prediction_prob}')

    # Display the image with the label
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f'Predicted: {label} ({prediction_prob:.2f})')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    model_path = 'plant_disease_model.h5'
    image_path = '../data/output/train-5.png'  # Update this path to a known healthy leaf
    classify_images(model_path, image_path)



