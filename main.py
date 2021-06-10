#!/usr/bin/env python3

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import scipy.io as spio
import time


BUFFER_SIZE = 60000
BATCH_SIZE = 31
kernel_length = 5
noise_dim = 100
max_input_value = 10000
mid_value = max_input_value / 2
dataset_file = "data/" + "data.mat"

train_dataset = spio.loadmat(dataset_file)

first_class_trainset = train_dataset["data"]

# Normalize data
first_class_trainset = (first_class_trainset - mid_value)/mid_value

first_class_trainset = first_class_trainset[:, :124]
first_class_trainset = np.transpose(first_class_trainset)
first_class_trainset = tf.convert_to_tensor(first_class_trainset, dtype=tf.float32)
(total_sample_size, num_bands) = first_class_trainset.shape
print("num bands =", num_bands)


def make_generator_model():

    model = tf.keras.Sequential()
    model.add(layers.Dense(num_bands * 256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((num_bands, 256)))
    assert model.output_shape == (None, num_bands, 256)  # Note: None is the batch size

    model.add(layers.Conv1DTranspose(128, kernel_length, strides=1, padding='same', use_bias=False))
    tf.print("output shape is " + str(model.output_shape))
    assert model.output_shape == (None, num_bands, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(64, kernel_length, strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, num_bands, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(1, kernel_length, strides=1, padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, num_bands, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, kernel_length, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(128, kernel_length, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)
print(decision)
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 2
num_examples_to_generate = 4

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        print("generated images shape is ", generated_images.shape)
        reshaped_images = tf.reshape(images, (images.shape[0], images.shape[1], 1))
        print("reshaped image shape is ", reshaped_images.shape)

        real_output = discriminator(reshaped_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        tf.print("epoch number is " + str(epoch))
        start = time.time()
        print("shape is ", dataset.shape)
        i = 0
        for batch in tf.data.Dataset.from_tensor_slices(dataset).batch(BATCH_SIZE):
            tf.print("batch shape is " + str(batch.shape))
            tf.print("batch number is " + str(i))
            i = i + 1
            train_step(batch)
        # Save the model every 1 epochs
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    checkpoint.save(file_prefix=checkpoint_prefix)
    generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    predictions = model(test_input, training=False)
    print((predictions[0] * mid_value) + mid_value)

    for i in range(predictions.shape[0]):
        plt.quiver(predictions[i, :, 0] * mid_value + mid_value)
        plt.show()
        # plt.subplot(4, 4, i+1)
        # plt.imshow(predictions[i, :, 0] * mid_value + mid_value, cmap='gray')
        # plt.axis('off')

    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


train(first_class_trainset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


#display_image(EPOCHS)


