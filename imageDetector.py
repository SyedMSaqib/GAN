import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32 * 32 * 3, activation='tanh'))
    model.add(Reshape((32, 32, 3)))
    return model


def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model


learning_rate = 0.0002
beta_1 = 0.5
optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1)


discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

generator = build_generator()


discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=optimizer)


def data_generator(batch_size):
    (X_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32')
    X_train = (X_train - 127.5) / 127.5  

    while True:
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)
        yield real_images, fake_images


def train_gan(generator, discriminator, gan, epochs, batch_size=128, checkpoint_interval=1000, start_epoch=0):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

  
    if os.path.exists('discriminator.h5'):
        discriminator = load_model('discriminator.h5')
        print("Discriminator model loaded.")
    if os.path.exists('generator.h5'):
        generator = load_model('generator.h5')
        print("Generator model loaded.")
        gan = Sequential([generator, discriminator])
        gan.compile(loss='binary_crossentropy', optimizer=optimizer)
        print("GAN model compiled.")

    for epoch in range(start_epoch, epochs):
        
        real_images, fake_images = next(data_generator(batch_size))

        
        d_loss_real = discriminator.train_on_batch(real_images, real)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")

 
        if epoch % checkpoint_interval == 0 and epoch > 0:
            generator.save('generator.h5')
            discriminator.save('discriminator.h5')
            print(f"Checkpoint saved at epoch {epoch}.")


train_gan(generator, discriminator, gan, epochs=10000, batch_size=128)



