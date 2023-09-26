"""
Author: Beichen Gao
15.06.2023

Script containing subclassed TF models
"""

import tensorflow as tf
from tensorflow import keras

class ResidualBlock1D(tf.keras.Model):
    def __init__(self, filter_num, dilation_rate, regularizer_term, initializer='he_normal'):
        super(ResidualBlock1D, self).__init__(name='')
        # First Convolution
        self.conv1 = tf.keras.layers.Conv1D(filter_num, 1, dilation_rate=dilation_rate, padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(regularizer_term),
                                            kernel_initializer=initializer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        # Second Convolution
        self.conv2 = tf.keras.layers.Conv1D(filter_num, 3, padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(regularizer_term),
                                            kernel_initializer=initializer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        # Additive Layer
        self.add = tf.keras.layers.Add()

    def call(self, inputs, training=False):
        # Forward pass through the layers

        # Shortcut connection
        shortcut = inputs

        # Batch Normalization and Activation
        x = self.bn1(inputs)
        x = tf.keras.layers.Activation('relu')(x)
        # First convolution
        x = self.conv1(x)
        # Second convolution (bottleneck)
        x = self.bn2(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = self.conv2(x)
        # Add shortcut connection
        x = self.add([x, shortcut])

        return x


class CNN_model_1D(keras.Model):

    def __init__(self, stride, filter_num, padding, kernel_size, dense_dim, dense_dropout, pool_size,
                 pool_stride, residual_blocks, dilation_rate, regularizer_term, initializer='he_normal'):
        super(CNN_model_1D, self).__init__()
        self.cnn1 = tf.keras.layers.Conv1D(strides=stride, filters=filter_num, kernel_size=kernel_size,
                                           padding=padding, kernel_initializer=initializer)
        self.res_blocks = []
        if residual_blocks != 0:
            dilation_rate = dilation_rate
            for i in range(residual_blocks):
                self.res_blocks.append(ResidualBlock1D(filter_num, dilation_rate, regularizer_term))
                dilation_rate = dilation_rate * (i + 1)
        else:
            self.res_blocks.append(tf.keras.layers.Activation('relu'))
        # Max Pooling and Dropout
        self.pool = tf.keras.layers.MaxPooling1D(pool_size, pool_stride)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(dense_dim, activation='relu', kernel_initializer=initializer)
        self.dropout1 = tf.keras.layers.Dropout(dense_dropout)
        self.classifier = tf.keras.layers.Dense(units=1, activation='sigmoid',
                                                kernel_regularizer=tf.keras.regularizers.l2(regularizer_term))

    @tf.function
    def call(self, inputs, training=False):
        # Initial convolution
        x = self.cnn1(inputs)
        for res_block in self.res_blocks:
            x = res_block(x)
        # Pool
        x = self.pool(x)
        x = self.dropout1(x)
        # Dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        # output
        return self.classifier(x)


class MLP(keras.Model):

    def __init__(self, dense_dim, dense_dropout, dense_layers, initializer='he_normal'):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dense_dim, activation='relu', kernel_initializer=initializer)
        self.dropout = tf.keras.layers.Dropout(dense_dropout)
        self.dense2 = tf.keras.layers.Dense(dense_dim / 2, activation='relu', kernel_initializer=initializer)
        self.classifier = tf.keras.layers.Dense(units=1, activation='sigmoid')
        self.dense_blocks = []
        if dense_layers != 0:
            for i in range(dense_layers):
                self.dense_blocks.append(
                    tf.keras.layers.Dense(dense_dim, activation='relu', kernel_initializer=initializer))
                self.dense_blocks.append(tf.keras.layers.Dropout(dense_dropout))
        else:
            self.dense_blocks.append(tf.keras.layers.Dropout(dense_dropout))

    @tf.function
    def call(self, inputs, training=False):
        # dense layers
        x = self.dense1(inputs)
        x = self.dropout(x)
        for dense_block in self.dense_blocks:
            x = dense_block(x)
        x = self.dense2(x)
        x = self.dropout(x)
        # final output
        return self.classifier(x)
