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
from model_def_modelparallel import get_custom_model

import smdistributed.modelparallel.tensorflow as smp
# SMP: Initialize
smp.init()

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
    
    train_dataset = get_dataset(training_dir+'/train.tfrecords',  batch_size)
    train_dataset = train_dataset.shuffle(10000)
    
    val_dataset = get_dataset(validation_dir+'/validation.tfrecords', batch_size)
    eval_dataset = get_dataset(eval_dir+'/eval.tfrecords', batch_size)
    
    # Load model
    model = get_custom_model(input_shape)
        
    # Optimizer
    if optimizer.lower() == 'adam':
        opt = Adam(lr=lr, decay=weight_decay)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(lr=lr, decay=weight_decay)
    else:
        opt = SGD(lr=lr, decay=weight_decay, momentum=momentum) 

    # Loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    # Metrics to track
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
    
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    
    # Get gradients
    @smp.step
    def get_grads(images, labels):
        train_pred = model(images, training=True)
        train_loss_value = loss_fn(labels, train_pred)

        grads = opt.get_gradients(train_loss_value, model.trainable_variables)
        return grads, train_loss_value, train_pred
    
    # Training step
    @tf.function
    def training_step(images, labels, first_batch):
        gradients, train_loss_value, train_pred = get_grads(images, labels)

        # SMP: Accumulate the gradients across microbatches
        gradients = [g.accumulate() for g in gradients]
        opt.apply_gradients(zip(gradients, model.trainable_variables))

        # SMP: Average the loss across microbatches
        train_loss(train_loss_value.reduce_mean())
        
        # SMP: Merge predictions across microbatches
        train_accuracy(labels, train_pred.merge())
        return train_loss_value.reduce_mean()

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
            
        print(f'Epoch: {epoch + 1}, '
              f'Epoch duration: {epoch_time} sec, '
              f'Training loss: {train_loss.result()}, '
              f'Training accuracy: {train_accuracy.result() * 100}')

    smp.barrier()
    print('====== End of training ======')

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
    parser.add_argument('--model-type',    type=str,   default='custom')

    # SageMaker parameters
    parser.add_argument('--model_dir',        type=str)
    parser.add_argument('--mp_parameters',    type=str)
    
    # Data directories and other options
    parser.add_argument('--train',            type=str,   default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation',       type=str,   default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--eval',             type=str,   default=os.environ['SM_CHANNEL_EVAL'])
    
    args = parser.parse_args()

    main(args)