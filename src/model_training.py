import os

from keras import Sequential
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.applications.resnet import ResNet50
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from data_preprocessing import get_data

def build_model(input_shape, num_classes):
    # Using ResNet50 pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=None, input_shape=input_shape)

    # Unfreeze only the top few layers of the model for fine-tuning
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data_dir = '../data'  # Update this path to your dataset location
    X_train, X_test, y_train, y_test, datagen = get_data(data_dir)

    input_shape = (128, 128, 3)
    num_classes = len(os.listdir(data_dir))
    model = build_model(input_shape, num_classes)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=25, validation_data=(X_test, y_test),
              callbacks=[early_stopping, reduce_lr])
    model.save('plant_disease_model.h5')

