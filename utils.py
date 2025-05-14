import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import matplotlib.pyplot as plt

MNIST_IMG_SHAPE = (28, 28, 1)
LATENT_DIM = 128
KERNEL_SIZE = 4

# Encode the image into the latent space of dimension LATENT_DIM
def build_encoder(img_shape, conditional=False, num_classes=10):
    inputs = Input(shape=img_shape)
    if conditional:
        labels = Input(shape=(num_classes,))
        x = layers.Concatenate()([x, labels])
        model_inputs = [inputs, labels]
    else:
        model_inputs = inputs

    x = layers.Conv2D(32, KERNEL_SIZE, strides=2, padding='same', activation='relu')(model_inputs)
    x = layers.Conv2D(64, KERNEL_SIZE, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, KERNEL_SIZE, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)

    z_mean = layers.Dense(LATENT_DIM)(x)
    z_log_var = layers.Dense(LATENT_DIM)(x)
    return Model(model_inputs, [z_mean, z_log_var], name='encoder')

# Sample from latent space the vector z = mean + sqrt(exp(log_var)) * epsilon
MAX_LOG_VAR = 10.0
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var = tf.clip_by_value(z_log_var, -MAX_LOG_VAR, MAX_LOG_VAR)
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Decode from the latent space the original image
def build_decoder(conditional=False, num_classes=10):
    latent_inputs = Input(shape=(LATENT_DIM,))
    if conditional:
        labels = Input(shape=(num_classes,))
        model_inputs = [latent_inputs, labels]
        concat_inputs = layers.Concatenate()(model_inputs)
    else:
        model_inputs = latent_inputs
    
    x = layers.Dense(7 * 7 * LATENT_DIM, activation='relu')(concat_inputs if conditional else latent_inputs)
    x = layers.Reshape((7, 7, LATENT_DIM))(x)
    x = layers.Conv2DTranspose(64, KERNEL_SIZE, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, KERNEL_SIZE, strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, KERNEL_SIZE-1, padding='same', activation='sigmoid')(x)
    return Model(model_inputs, outputs, name='decoder')

# VAE class that has an encoder, a sampler, a decoder and the VAE loss
class VAE(Model):

    def __init__(self, encoder, decoder, conditional=False, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampling()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.conditional = conditional

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.recon_loss_fn = tf.keras.losses.MeanSquaredError()
        
    def reconstruct_images(self, inputs, batch_size=None):
        if self.conditional:
            images, labels = inputs
            z_mean, z_log_var = self.encoder.predict([images, labels], batch_size=batch_size)
            z = self.sampler([z_mean, z_log_var])
            recon_images = self.decoder.predict([z, labels], batch_size=batch_size)
        else:
            images = inputs
            z_mean, z_log_var = self.encoder.predict(images, batch_size=batch_size)
            z = self.sampler([z_mean, z_log_var])
            recon_images = self.decoder.predict(z, batch_size=batch_size)
        return recon_images
        
    def train_step(self, data):
        if self.conditional:
            data, label = data
    
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder([data, label] if self.conditional else data)
            z = self.sampler([z_mean, z_log_var])
            decoded_image = self.decoder([z, label] if self.conditional else z)
            loss = self.vae_loss(data, decoded_image, (z_mean, z_log_var))
    
        grads = tape.gradient(loss, self.trainable_weights)
        for g in grads:
            tf.debugging.check_numerics(g, message="Gradient NaN detected")
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)
        return {
            "loss": self.total_loss_tracker.result()
        }

    def vae_loss(self, data, decoded_image, encoder_output):
        gamma = 0.0001
        z_mean, z_log_var = encoder_output
        recon_loss = (self.recon_loss_fn(data, decoded_image))
        kl_loss = -0.5 * tf.keras.backend.sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        kl_loss = tf.reduce_mean(kl_loss)
        return recon_loss + gamma * kl_loss

# Show real images reconstruction
def show_images_reconstruction(vae_model: VAE, x_test, label=None, n=16, plot=True):
    real_images = x_test[:n]
    if label is not None:
        # Conditional VAE
        real_labels = label[:n]
        recon_images = vae_model.reconstruct_images((real_images, real_labels))
    else:
        # VAE
        recon_images = vae_model.reconstruct_images(real_images)

    if plot:
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(real_images[i])
            plt.axis("off")

            # Reconstructed
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(recon_images[i])
            plt.axis("off")

    plt.suptitle("Top: Original | Bottom: Reconstruction")
    plt.show()

    return recon_images
