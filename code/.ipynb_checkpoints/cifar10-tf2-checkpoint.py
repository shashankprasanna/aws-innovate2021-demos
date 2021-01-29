import tensorflow as tf

from datetime import datetime
import argparse
import os
import numpy as np
import codecs
import json
import boto3
import time

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from model_def import get_custom_model

HEIGHT = 32
WIDTH  = 32
DEPTH  = 3
input_shape = (HEIGHT, WIDTH, DEPTH)
NUM_CLASSES = 10
NUM_TRAIN_IMAGES = 40000
NUM_VALID_IMAGES = 10000
NUM_TEST_IMAGES  = 10000

def single_example_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.io.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)
    
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)
    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [HEIGHT, WIDTH, DEPTH])
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def get_dataset(filenames, batch_size):
    """Read the images and labels from 'filenames'."""
    # Load dataset.
    dataset = tf.data.TFRecordDataset(filenames)

    # Parse records.
    dataset = dataset.map(single_example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def get_model(model_type):
    print(f'====== Getting model architecture: {model_type} ======')
    input_tensor = Input(shape=input_shape)
    if model_type == 'resnet':
        base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                          weights='imagenet',
                                                          input_tensor=input_tensor,
                                                          input_shape=input_shape,
                                                          classes=None)
        x = Flatten()(base_model.output)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    elif model_type == 'vgg':
        base_model = keras.applications.vgg19.VGG19(include_top=False,
                                                          weights=None,
                                                          input_tensor=input_tensor,
                                                          input_shape=input_shape,
                                                          classes=None)
        x = Flatten()(base_model.output)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    else:
        model = get_custom_model(input_shape)
        
    return model

        
def main(args):
    # Hyper-parameters
    epochs       = args.epochs
    lr           = args.learning_rate
    batch_size   = args.batch_size
    momentum     = args.momentum
    weight_decay = args.weight_decay
    optimizer    = args.optimizer
    model_type   = args.model_type

    # SageMaker options
    training_dir     = args.train
    validation_dir   = args.validation
    eval_dir         = args.eval
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Get dataset
    train_dataset = get_dataset(training_dir+'/train.tfrecords',  batch_size)
    train_dataset = train_dataset.shuffle(10000)
    
    val_dataset = get_dataset(validation_dir+'/validation.tfrecords', batch_size)
    eval_dataset = get_dataset(eval_dir+'/eval.tfrecords', batch_size)
    
    # Load model
    model = get_model(model_type)
    
    # Optimizer
    if optimizer.lower() == 'adam':
        opt = Adam(lr=lr, decay=weight_decay)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(lr=lr, decay=weight_decay)
    else:
        opt = SGD(lr=lr, decay=weight_decay, momentum=momentum) 

    # Loss function
    loss = tf.keras.losses.CategoricalCrossentropy()
    
    # Metric to track
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
    
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    
    # Training step
    @tf.function
    def training_step(images, labels, first_batch):
        with tf.GradientTape() as tape:
            train_pred = model(images, training=True)
            train_loss_value = loss(labels, train_pred)

        grads = tape.gradient(train_loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(train_loss_value)
        train_accuracy(labels, train_pred)
        return
    
    # Testing step
    @tf.function
    def test_step(images, labels):
        val_pred = model(images, training=False)
        val_loss_value = loss(labels, val_pred)
        
        val_loss(val_loss_value)
        val_accuracy(labels, val_pred)
        return

    # Training loop
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
        
        for batch, (images, labels) in enumerate(train_dataset):
            start_time = time.time()
            training_step(images, labels, batch == 0)
            epoch_time = time.time() - start_time
            
        for images, labels in val_dataset:
            test_step(images, labels)
        
        print(f'Epoch: {epoch + 1}, '
              f'Epoch duration: {epoch_time} sec, '
              f'Training loss: {train_loss.result()}, '
              f'Training accuracy: {train_accuracy.result() * 100}',
              f'Validation Loss: {val_loss.result()}, '
              f'Validation Accuracy: {val_accuracy.result() * 100}')

    for images, labels in eval_dataset:
        test_pred = model(images, training=False)
        test_loss_value = loss(labels, test_pred)
        
        test_loss(test_loss_value)
        test_accuracy(labels, test_pred)
    
    print('====== Test Results ======')
    print(f'Test loss: {test_loss.result()}, '
          f'Test accuracy: {test_accuracy.result() * 100}')
    print('====== End of training ======')
    
    model.save(os.path.join(os.environ["SM_MODEL_DIR"], '1'))

if __name__ == "__main__":
    
    # Change: Update script to accept hyperparameters as command line arguments
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--epochs',        type=int,   default=15)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size',    type=int,   default=256)
    parser.add_argument('--weight-decay',  type=float, default=2e-4)
    parser.add_argument('--momentum',      type=float, default='0.9')
    parser.add_argument('--optimizer',     type=str,   default='adam')
    parser.add_argument('--model-type',    type=str,   default='resnet')

    # SageMaker parameters
    parser.add_argument('--model_dir',        type=str)
    
    # Data directories and other options
    parser.add_argument('--train',            type=str,   default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation',       type=str,   default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--eval',             type=str,   default=os.environ['SM_CHANNEL_EVAL'])
    
    args = parser.parse_args()

    main(args)