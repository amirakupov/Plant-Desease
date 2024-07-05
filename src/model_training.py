import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preprocessing import get_data

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')


def build_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers[-4:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', dtype='float32')  # Output layer should be float32
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    data_dir = '../data'
    batch_size = 32
    train_dataset, test_dataset, class_weights = get_data(data_dir, batch_size)

    input_shape = (96, 96, 3)  # Ensure the input shape matches the resized image size
    num_classes = 2  # Healthy and Diseased
    model = build_model(input_shape, num_classes)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    model.fit(train_dataset, epochs=25, validation_data=test_dataset,
              callbacks=[early_stopping, reduce_lr], class_weight=class_weights)
    model.save('plant_disease_model.h5')
