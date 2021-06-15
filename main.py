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
# dataset_file = "data/" + "target_based_samples.mat"
checkpoint_dir = './training_checkpoints'
num_examples_to_generate = 5
IMAGE_DIR = "img/"

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

seed = tf.random.normal([num_examples_to_generate, noise_dim])

# classes_array = [124,
#                  64,
#                  147,
#                  129,
#                  193
#                  ]

# classes_array = [6, 18, 11, 1, 5]

# train_dataset = spio.loadmat(dataset_file)
# total_trainset = train_dataset["data"]


def normalize_data(data):
    max_input = np.max(data)
    mid_input = max_input/2
    return (data - mid_input) / mid_input


def denormalize_data(normalized_data):
    max_input = np.max(normalized_data)
    mid_input = max_input/2
    return (normalized_data * mid_input) + mid_input


# Normalize data
#normalized_input_data = normalize_data(total_trainset)
num_bands = 0
#print("number of bands = ", num_bands)


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
classifier_optimizer = tf.keras.optimizers.Adam(1e-4)


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
        for batch in tf.data.Dataset.from_tensor_slices(dataset).shuffle(20).batch(BATCH_SIZE):
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
        y = denormalize_data(predictions[i])
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
        y = denormalize_data(before_tensor_input[i])
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


def make_classifier_model(num_classes):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, 5, strides=1, padding='same', use_bias=False, kernel_regularizer=None))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(128, 5, strides=1, padding='same', use_bias=False, kernel_regularizer=None))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(72, 5, strides=1, padding='same', use_bias=False, kernel_regularizer=None))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, use_bias=False, kernel_regularizer=None))
    return model


@tf.function
def classifier_train_step(train_features, train_labels, model):
    with tf.GradientTape() as tape:
        train_features_reshaped = np.reshape(train_features, (train_features.shape[0], train_features.shape[1], 1))
        #tf.print("reshaped size: "+ str(train_features_reshaped.shape))
        logits = model(train_features_reshaped, training=True)  # Logits for this minibatch
        # Compute the loss value for this minibatch.
        #print('train features shape: ', train_features.shape)
        #print('training labels shapes: ', train_labels.shape)
        #print('logits shapes: ', logits.shape)
        loss_value = classifier_loss_fn(train_labels, logits)
    gradients_of_classifier = tape.gradient(loss_value, model.trainable_variables)
    classifier_optimizer.apply_gradients(zip(gradients_of_classifier, model.trainable_variables))


def classifier_train(dataset, epochs, model, checkpoint, checkpoint_prefix_classifier):
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
    #checkpoint.save(file_prefix=checkpoint_prefix_classifier)
    model.save_weights(filepath=checkpoint_prefix_classifier)


def start_train(normalized_input_data, classes_array, epochs):
    global num_bands
    num_bands = normalized_input_data.shape[0]
    start = 0
    generated_samples = np.empty(shape=(0, num_bands))
    for i, number_of_samples_in_class in enumerate(classes_array):
        print("class:", i)
        end = start + number_of_samples_in_class
        #print("using start=", start, "end=", end)
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

        generated_samples = train(class_trainset, epochs, before_tensor_input, generator, discriminator, checkpoint_prefix, checkpoint,
                                  generated_samples)
        start = start + number_of_samples_in_class

    print("GAN training is done, total generated samples count:", str(generated_samples.shape[0]))
    print("")
    print("***** Starting Classification *****")
    label_columns = np.zeros(shape=(num_examples_to_generate * len(classes_array), len(classes_array)))
    k = 0
    for class_index in range(len(classes_array)):
        for i in range(num_examples_to_generate):
            label_columns[k][class_index] = 1
            k = k + 1
    labelled_generated_samples = np.hstack((generated_samples, label_columns))
    # print("shape after resampling: ", labelled_generated_samples.shape)
    # print(generated_samples.shape)
    #print(label_columns)
    model = make_classifier_model(len(classes_array))
    classifier_checkpoint = tf.train.Checkpoint(classifier_optimizer=classifier_optimizer,
                                                model=model)
    classifier_train(labelled_generated_samples, epochs, model, classifier_checkpoint,
                     get_classifier_checkpoint_prefix(epochs))

    final_evaluation_data = np.transpose(normalized_input_data)
    label_columns = np.zeros(shape=(final_evaluation_data.shape[0], len(classes_array)))
    k = 0
    for class_index, samples_in_class in enumerate(classes_array):
        for i in range(samples_in_class):
            label_columns[k][class_index] = 1
            k = k + 1
    evluation_data_reshaped = np.reshape(final_evaluation_data, (final_evaluation_data.shape[0], final_evaluation_data.shape[1], 1))
    prediction = model(evluation_data_reshaped, training=False)
    predictions_file = 'prediction_' + str(epochs) + '.csv'
    np.savetxt(predictions_file, prediction, fmt='%5.10f', delimiter=',')
    print("Saved predictions in the file:", predictions_file)
    # Print prediction for first 5 rows
    #print(prediction[:5, :])
    final_loss = classifier_loss_fn(label_columns, prediction)
    print("final loss of classifier with %d epochs is %3.10f" % (epochs, final_loss.numpy()))


def reload_classifier(dataset_name, dataset, num_classes, pred_epochs_count):
    print()
    print('***** Starting prediction of dataset %s by reloading weights *****' % dataset_name)
    checkpoint_pref = get_classifier_checkpoint_prefix(pred_epochs_count)
    print("using weights saved in", checkpoint_pref)
    dataset = np.transpose(normalize_data(dataset))
    dataset = np.reshape(dataset, (dataset.shape[0], dataset.shape[1], 1))
    model = make_classifier_model(num_classes)
    model.load_weights(filepath=checkpoint_pref).expect_partial()
    new_pred = model(dataset, training=False)
    pred_reload_file = 'prediction_reload_' + dataset_name + '_' + str(pred_epochs_count) + '.csv'
    np.savetxt(pred_reload_file, new_pred, fmt='%5.10f', delimiter=',')
    print("Saved predictions in", pred_reload_file)
    return new_pred


def main(dataset_name, classes_array=None, training_epochs=1000, pred_epochs_count=1000, num_classes=None):
    """

    :param training_epochs:
    :param num_classes:
    :param dataset_name:
    :param classes_array:
    :param pred_epochs_count:
    :return:
    """
    dataset_file_name = "data/" + dataset_name + ".mat"
    print("loading data from", dataset_file_name)
    training_dataset = spio.loadmat(dataset_file_name)
    training_data = training_dataset["data"]
    normalised_data = normalize_data(training_data)
    if classes_array is not None:
        print("Classes array is provided. Doing training")
        print("Number of epochs:", training_epochs)
        start_train(normalised_data, classes_array, training_epochs)
    else:
        print("Classes array not provided hence doing classification by reloading weights.")
        if num_classes is None:
            raise ValueError("num_classes is not provided for reload classifier operation")
        reload_classifier(dataset_name, normalised_data, num_classes, pred_epochs_count)


def get_classifier_checkpoint_prefix(epochs):
    return os.path.join(checkpoint_dir, "classifier_" + str(epochs))


if __name__ == "__main__":
    main("data", [124, 64, 147, 129, 193], training_epochs=1)
    main("target_based_samples", pred_epochs_count=1, num_classes=5)

    #                  ])
    # dataset=
    # reload_classifier(dataset)

#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

# display_image(EPOCHS)
