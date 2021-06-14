#!/usr/bin/env python3

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import scipy.io as spio
import time

tf.config.run_functions_eagerly(True)

BATCH_SIZE = 10
CLASSIFIER_BATCH_SIZE = 15
kernel_length = 5
noise_dim = 100
dataset_file = "data/" + "target_based_samples.mat"
checkpoint_dir = './training_checkpoints'
EPOCHS = 2
num_examples_to_generate = 500
IMAGE_DIR = "img/"

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

seed = tf.random.normal([num_examples_to_generate, noise_dim])

# classes_array = [124,
#                  64,
#                  147,
#                  129,
#                  193
#                  ]

classes_array = [6, 18, 11, 1, 5]

train_dataset = spio.loadmat(dataset_file)
total_trainset = train_dataset["data"]


def normalize_data(data, mid_value):
    return (data - mid_value) / mid_value


def denormalize_data(normalized_data, mid_value):
    return (normalized_data * mid_value) + mid_value


# Normalize data
max_input_value = np.max(total_trainset)
mid_value = max_input_value / 2
normalized_input_data = normalize_data(total_trainset, mid_value)
num_bands = normalized_input_data.shape[0]
print("mid value ", mid_value)
print("number of bands = ", num_bands)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(num_bands * 256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((num_bands, 256)))
    assert model.output_shape == (None, num_bands, 256)  # Note: None is the batch size

    model.add(layers.Conv1DTranspose(128, kernel_length, strides=1, padding='same', use_bias=False))
    # tf.print("output shape is " + str(model.output_shape))
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


# This method returns a helper function to compute cross entropy loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        # print("generated images shape is ", generated_images.shape)
        reshaped_images = tf.reshape(images, (images.shape[0], images.shape[1], 1))
        # print("reshaped image shape is ", reshaped_images.shape)

        real_output = discriminator(reshaped_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs, before_tensor_input, generator, discriminator, checkpoint_prefix, checkpoint, generated_samples):
    # generate_and_save_images(generator, epochs, seed, before_tensor_input, checkpoint_prefix+'_before')
    for epoch in range(epochs):
        #tf.print("epoch number is " + str(epoch))
        start = time.time()
        i = 0
        for batch in tf.data.Dataset.from_tensor_slices(dataset).batch(BATCH_SIZE):
            # tf.print("batch shape is " + str(batch.shape))
            #tf.print("batch number is " + str(i))
            i = i + 1
            train_step(batch, generator, discriminator)
        # Save the model every 1 epochs
        tf.print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    checkpoint.save(file_prefix=checkpoint_prefix)
    return generate_and_save_images(generator, epochs, seed, before_tensor_input, checkpoint_prefix+'_after', generated_samples)


def generate_and_save_images(model, epoch, test_input, before_tensor_input, checkpoint_prefix, generated_samples):
    # Notice `training` is set to False.
    predictions = model(test_input, training=False)
    # print('size of generated_samples is ', generated_samples.shape)
    checkpoint_prefix = checkpoint_prefix.split("/")[-1]
    plt.figure()
    for i in range(min(predictions.shape[0], 4)):
        y = denormalize_data(predictions[i], mid_value)
        x = range(predictions.shape[1])
        plt.plot(x, y, label="genereted image " + checkpoint_prefix + str(i))
        # plt.subplot(4, 4, i+1)
        # plt.imshow(predictions[i, :, 0] * mid_value + mid_value, cmap='gray')
        # plt.axis('off')
    plt.legend()
    plt.savefig(IMAGE_DIR + 'generated_' + checkpoint_prefix + '.png')

    plt.figure()
    for i in range(min(predictions.shape[0], 1)):
        x = range(predictions.shape[1])
        y = denormalize_data(before_tensor_input[i], mid_value)
        plt.plot(x, y, label="original input sample" + checkpoint_prefix + str(i))
    plt.legend()
    plt.savefig(IMAGE_DIR + 'original_' + checkpoint_prefix + '.png')

    pred_shape = predictions.shape
    predictions = predictions.numpy()
    predictions = predictions.reshape(pred_shape[0], pred_shape[1])
    # print("predictions shape is ", predictions.shape)
    # print("generated samples shape", generated_samples.shape)
    generated_samples = np.concatenate((generated_samples, predictions))
    return generated_samples
    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


classifier_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


def make_classifier_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(20))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Conv1DTranspose(15, kernel_length, strides=1, padding='same', use_bias=False))
    # model.add(layers.Conv1DTranspose(10, kernel_length, strides=1, padding='same', use_bias=False, activation='tanh'))
    model.add(layers.Dense(len(classes_array)))

    return model

@tf.function
def classifier_train_step(train_features, train_labels, model):
    with tf.GradientTape() as tape:
        logits = model(train_features, training=True)  # Logits for this minibatch
        # Compute the loss value for this minibatch.
        #print('train features shape: ', train_features.shape)
        #print('training labels shapes: ', train_labels.shape)
        #print('logits shapes: ', logits.shape)
        loss_value = classifier_loss_fn(train_labels, logits)
    gradients_of_classifier = tape.gradient(loss_value, model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_classifier, model.trainable_variables))


def classifier_train(dataset, epochs, model):
    for epoch in range(epochs):
        # tf.print("epoch number is " + str(epoch))
        start = time.time()
        i = 0
        for batch in tf.data.Dataset.from_tensor_slices(dataset).batch(CLASSIFIER_BATCH_SIZE):
            # tf.print("batch shape is " + str(batch.shape))
            #tf.print("batch number is " + str(i))
            i = i + 1
            classifier_train_step(batch[:, :num_bands], batch[:, num_bands:], model)
        # Save the model every 1 epochs
        tf.print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def start_train():
    start = 0
    generated_samples = np.empty(shape=(0, num_bands))
    for i, number_of_samples_in_class in enumerate(classes_array):
        print("class:", i)
        end = start + number_of_samples_in_class
        print("using start=", start, "end=", end)
        class_trainset = normalized_input_data[:, start:end]
        class_trainset = np.transpose(class_trainset)
        before_tensor_input = np.copy(class_trainset)
        class_trainset = tf.convert_to_tensor(class_trainset, dtype=tf.float32)

        generator = make_generator_model()
        discriminator = make_discriminator_model()
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_" + str(i))
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)

        generated_samples = train(class_trainset, EPOCHS, before_tensor_input, generator, discriminator, checkpoint_prefix, checkpoint,
                                  generated_samples)
        start = start + number_of_samples_in_class
        #print("total generated_sample shape", generated_samples.shape)

    label_columns = np.zeros(shape=(num_examples_to_generate * len(classes_array), len(classes_array)))
    k = 0
    for class_index in range(len(classes_array)):
        for i in range(num_examples_to_generate):
            label_columns[k][class_index] = 1
            k = k + 1
    labelled_generated_samples = np.hstack((generated_samples, label_columns))
    #print("shape after resampling: ", labelled_generated_samples.shape)
    #print(labelled_generated_samples)
    model = make_classifier_model()
    classifier_train(labelled_generated_samples, EPOCHS, model)

    final_evaluation_data = np.transpose(normalized_input_data)
    #print("normalised input shape", final_evaluation_data.shape)
    label_columns = np.zeros(shape=(final_evaluation_data.shape[0], len(classes_array)))
    k = 0
    for class_index, samples_in_class in enumerate(classes_array):
        for i in range(samples_in_class):
            label_columns[k][class_index] = 1
            k = k + 1
    prediction = model(final_evaluation_data)
    np.savetxt('prediction_' + str(EPOCHS) + '.csv', prediction, fmt='%5.10f')
    # Print prediction for first 5 rows
    #print(prediction[:5, :])
    final_loss = classifier_loss_fn(label_columns, prediction)
    print("final loss with %d epochs is %3.10f" % (EPOCHS, final_loss.numpy()))


def main():
    start_train()


if __name__ == "__main__":
    main()




#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

# display_image(EPOCHS)
