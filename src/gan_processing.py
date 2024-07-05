import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, LeakyReLU, Conv2D, BatchNormalization, Activation, Dropout, Flatten, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import RandomNormal, GlorotUniform
from tensorflow.keras.initializers import lecun_normal
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import csv

def init_csv(log_file_path='training_log.csv'):
    with open(log_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Discriminator Loss", "Generator Loss", "Accuracy"])

def log_metrics_to_csv(epoch, d_loss, g_loss, log_file_path='training_log.csv'):
    with open(log_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, d_loss, g_loss[0], g_loss[2]])

# Definiere den Generator mit verbesserten Initialisierungen
def build_generator():
    gen_initializer = GlorotUniform()
    model = tf.keras.Sequential([
        Input(shape=(100,)),
        Dense(256 * 8 * 8, activation="relu", kernel_initializer=gen_initializer),
        Reshape((8, 8, 256)),
        UpSampling2D(),
        Conv2D(256, kernel_size=3, padding="same"),
        BatchNormalization(momentum=0.8),
        Activation("relu"),
        UpSampling2D(),
        Conv2D(128, kernel_size=3, padding="same"),
        BatchNormalization(momentum=0.8),
        Activation("relu"),
        Conv2D(64, kernel_size=3, padding="same"),
        BatchNormalization(momentum=0.8),
        Activation("relu"),
        Conv2D(3, kernel_size=3, padding="same"),
        Activation("tanh")
    ])
    return model

# Definiere den Diskriminator mit einer verbesserten Initialisierung
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(Input(shape=(64, 64, 3)))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())

    # Print the output shape after flattening
    print("Output shape after flattening:", model.output_shape)

    model.add(Dense(1, activation='sigmoid'))
    return model

# Definiere das GAN
def build_gan(generator, discriminator):
    model = tf.keras.Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Lade die Bilder aus dem data Verzeichnis
def load_real_data(data_dir='../data'):
    data = []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                data.append(img)
    data = np.array(data)
    data = (data.astype(np.float32) - 127.5) / 127.5  # Normalisierung auf den Bereich [-1, 1]
    return data

def load_model_if_exists(model_path):
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        return None

# Training des GAN
def train_gan(generator, discriminator, gan, start_epoch=2863, epochs=1000, batch_size=32, save_interval=500):
    discriminator.trainable = False
    X_train = load_real_data()
    half_batch = batch_size // 2
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_images = X_train[idx]
        noise = np.random.normal(0, 1, (half_batch, 100))
        fake_images = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)
        print(f"Epoch: {epoch}")
        if epoch % save_interval == 0:
            generate_and_save_images(generator, epoch+start_epoch)
            log_metrics_to_csv(epoch+start_epoch, d_loss[0], g_loss)
        if epoch % (save_interval/5) == 0:
            generator.save('gan_generator_2.h5')
            discriminator.save('gan_discriminator_2.h5')

def generate_and_save_images(generator, epoch, num_images=16, image_size=(64, 64), grid_size=(4, 4), save_to_disk=True, save_path='../data/gan_images'):
    noise = np.random.normal(0, 1, (num_images, 100))
    gen_images = generator.predict(noise)
    gen_images = 0.5 * gen_images + 0.5  # Rückskalieren auf den Bereich [0, 1]
    os.makedirs(save_path, exist_ok=True)
    fig, axs = plt.subplots(grid_size[0], grid_size[1])
    cnt = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            axs[i, j].imshow(gen_images[cnt, :, :, :])
            axs[i, j].axis('off')
            cnt += 1
    plt.show()
    if save_to_disk:
        for i in range(num_images):
            plt.imsave(f'{save_path}/generated_image_{epoch}_{i}.png', gen_images[i, :, :, :])

if __name__ == "__main__":
    optimizer = Adam(0.0002, 0.5)
    discriminator = load_model_if_exists('gan_discriminator_2.h5')
    if discriminator is None:
        discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    generator = load_model_if_exists('gan_generator_2.h5')
    if generator is None:
        generator = build_generator()
    gan = build_gan(generator, discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    train_gan(generator, discriminator, gan)
