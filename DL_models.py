"""
Authors: Beichen Gao, Jiami Han

Main script for training, testing, and HP optimization of MLP and CNN models.

Code is written to take input arguments compatible with Slurm Workload Manager on ETH Euler clusters. Can also be changed to run
locally and take local arguments
"""
import argparse
import os
import pandas as pd
import numpy as np
import math
import time
from datetime import datetime
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score, auc, \
    roc_auc_score
import keras.backend as K
from helpers import encode_onehot_padded
import random


def create_parser():
    parser = argparse.ArgumentParser(description="Training of deep learning models for omicron RBD", fromfile_prefix_chars='@')

    # ---data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='location of labelled datasets for model training')
    parser.add_argument('--target', type=str, default='ACE2',
                        help='location of labelled datasets for model training')
    parser.add_argument('--library', type=str, default='Lib1',
                        help='name of the dataset used for training')
    parser.add_argument('--count_threshold', type=int, default=2,
                        help='threshold of counts to use for training sequences. Removes sequences below threshold')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for run')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='used for train_test_split, test size')
    parser.add_argument('--val_size', type=float, default=0.11,
                        help='used for train_test_split, validation size')
    parser.add_argument('--rejection_sampling', type=str, default='true',
                        help='whether to perform balancing or not. Options: none, true')
    parser.add_argument('--minority_ratio', type=float, default=0.25,
                        help='ratio of minority class to majority for rejection sampling')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='patience for early_stopping during training')
    parser.add_argument('--test_variants', type=str, default='VOC_Spike_all',
                        help='path to .csv containing variants to test predictions on (VoC or synthetic)')
    parser.add_argument('--test_variants_colname', type=str, default='sequence',
                        help='name of column containing sequences of variants to test predictions on (VoC or synthetic)')
    # ---DL arguments
    parser.add_argument("--base_model", type=str, default='cnn1d',
                        help='Type of deep learning model to use. Options: cnn1d, mlp')
    parser.add_argument('--embedding', type=str, default='onehot',
                        help='embedding to use for sequences, options: onehot (esm removed)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size during training')
    parser.add_argument('--epoch', type=int, default=30,
                        help='epochs to train model')
    parser.add_argument('--learn_rate', type=float, default=0.001,
                        help='learning rate for Adam optimizer')
    parser.add_argument('--regularizer_term', type=float, default=0.001,
                        help='kernel regularizer term')
    # ---CNN arguments
    parser.add_argument("--residual_blocks", type=int, default=0, help='Number of residual blocks in 1D network')
    parser.add_argument("--dilation_rate", type=int, default=2,
                        help='Rate of increasing dilations between residual blocks')
    parser.add_argument("--kernel_size", type=int, default=9, help='Size of the convolution kernel')
    parser.add_argument("--filter_num", type=int, default=32, help='Number of filters in the convolution layer')
    parser.add_argument("--stride", type=int, default=1, help='Stride of the convolution operation')
    parser.add_argument("--padding", type=str, default="same", help='Padding method for the convolution operation')
    parser.add_argument("--pool_size", type=int, default=2, help='pool_size in Maxpooling layer')
    parser.add_argument("--pool_stride", type=int, default=2, help='Stride of the pooling operation')

    # ---Dense arguments
    parser.add_argument("--dense_dim", type=int, default=32, help='Number of neuron in dense layer')
    parser.add_argument('--dense_dropout', type=float, default=0.2,
                        help='dropout for dense layer')
    parser.add_argument('--dense_layers', type=int, default=1,
                        help='number of additional dense layers in MLP')

    return parser

class ResidualBlock1D(tf.keras.Model):
    def __init__(self, filter_num, dilation_rate, regularizer_term):
        super(ResidualBlock1D, self).__init__(name='')
        # First Convolution
        self.conv1 = tf.keras.layers.Conv1D(filter_num, 1, dilation_rate=dilation_rate, padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(regularizer_term))
        self.bn1 = tf.keras.layers.BatchNormalization()
        # Second Convolution
        self.conv2 = tf.keras.layers.Conv1D(filter_num, 3, padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(regularizer_term))
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
                 pool_stride, residual_blocks, dilation_rate, regularizer_term):
        super(CNN_model_1D, self).__init__()
        self.cnn1 = tf.keras.layers.Conv1D(strides=stride, filters=filter_num, kernel_size=kernel_size, padding=padding)
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
        self.dense1 = tf.keras.layers.Dense(dense_dim, activation='relu')
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


def encode_data(aa, encoding):
    """
        Encodes the data.
        Input:
            aa: list of aa sequences
    """
    x = []
    if encoding == "onehot":
        x = encode_onehot_padded(aa)
    return x


# ==================================================================
# ======================= Model full run ===========================
# ==================================================================

def main(args):

    WORK_DIR = os.getcwd()
    DATA_DIR = f"{WORK_DIR}/{args.data_dir}"

    # set random_seeds
    if args.seed == 0:
        seed_list = [1, 2, 3]
    else:
        seed_list = [args.seed]
    # ======================= Track best performances =======================

    for seed_entry in seed_list:
        # set the seeds
        np.random.seed(seed_entry)
        tf.random.set_seed(seed_entry)
        print(f'SEED: {seed_entry}')
        # get time string for tracking runs
        now = datetime.now()
        timestr = now.strftime("%Y%m%d-%H%M-%f")

        # ======================= Print all relevant args =======================
        print(f'Running with the {args.target}_{args.library} dataset with the following parameters:')
        print(f'Model Type: {args.base_model}')
        print(f'Count Threshold: {str(args.count_threshold)}')
        print(f'Batch Size: {str(args.batch_size)}')
        print(f'Epochs: {str(args.epoch)}')
        print(f'Stride: {str(args.stride)}')
        print(f'Filter Num: {str(args.filter_num)}')
        print(f'Dense Dimension: {str(args.dense_dim)}')
        print(f'Padding: {str(args.padding)}')
        print(f'Kernel Size: {str(args.kernel_size)}')
        # create run_name and model_name variables for all future naming
        run_name = f'{args.target}_{args.library}_{seed_entry}_{args.base_model}_thres{args.count_threshold}_min{args.minority_ratio}_{timestr}'

        # ======================= Load Datasets =======================
        data = pd.read_csv(
            f"{DATA_DIR}/{args.library}/{args.target}_{args.library}_labeled.csv",
            index_col=0)
        # clean dataset
        data = data[data['Total_sum'] >= args.count_threshold]

        # ======================= Split up datasets =======================
        # split into train, test, validation sets
        train, test = train_test_split(data, test_size=args.test_size)
        train, val = train_test_split(train, test_size=args.val_size)

        # ======================= Encode Sequences =======================
        # encode sequences
        train_x = encode_data(train.aa, encoding=args.embedding)
        test_x = encode_data(test.aa, encoding=args.embedding)
        val_x = encode_data(val.aa, encoding=args.embedding)
        if args.base_model == 'mlp':
            # flatten encodings
            train_x = train_x.reshape(train_x.shape[0], -1)
            test_x = test_x.reshape(test_x.shape[0], -1)
            val_x = val_x.reshape(val_x.shape[0], -1)

        train_y = train.Label
        test_y = test.Label
        val_y = val.Label

        # ======================= Create DataLoaders =======================
        def class_func(features, label):
            return label

        # Training set first
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_dataset = train_dataset.repeat().shuffle(buffer_size=256).batch(args.batch_size)
        # Create balanced dataset if rejection_sampling == true
        if args.rejection_sampling == 'true':
            # get the ratios
            majority_ratio = 1 - args.minority_ratio
            # find majority class
            if train_y.value_counts().sort_values().index[-1] == 0:
                resample_ds = (train_dataset.unbatch().rejection_resample(class_func, target_dist=[majority_ratio,
                                                                                                   args.minority_ratio]).batch(
                    args.batch_size))
            else:
                resample_ds = (train_dataset.unbatch().rejection_resample(class_func, target_dist=[args.minority_ratio,
                                                                                                   majority_ratio]).batch(
                    args.batch_size))
            train_dataset = resample_ds.map(lambda extra_label, features_and_label: features_and_label)

        # then validation
        val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        val_dataset = val_dataset.batch(args.batch_size)

        # ======================= Set model parameters =======================
        # Custom MCC
        def binary_mcc(y_true, y_pred):
            tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
            fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
            fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

            num = tp * tn - fp * fn
            den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            return num / K.sqrt(den + K.epsilon())

        loss_fn = keras.losses.BinaryCrossentropy()
        opt = keras.optimizers.Adam(learning_rate=args.learn_rate)
        model_metrics = [keras.metrics.BinaryAccuracy(),
                        tfa.metrics.F1Score(num_classes=1, average='macro', threshold=0.5),
                         keras.metrics.AUC(curve='ROC', name='roc_auc'),
                         keras.metrics.AUC(curve='PR', name='pr_auc'),
                         binary_mcc]

        # calculate training steps
        train_steps = math.floor(len(train) / args.batch_size)
        # callbacks
        LOG_DIR = f'{WORK_DIR}/logs/{args.target}'
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'{LOG_DIR}/{run_name}/', histogram_freq=2)
        if args.early_stop != 0:
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.early_stop)
            callbacks = [tensorboard_callback, early_stop]
        else:
            callbacks = [tensorboard_callback]

        # ======================= Create model  =======================
        if args.base_model == 'cnn1d':
            model = CNN_model_1D(stride=args.stride, filter_num=args.filter_num, padding=args.padding,
                                 kernel_size=args.kernel_size, dense_dim=args.dense_dim,
                                 dense_dropout=args.dense_dropout,
                                 pool_size=args.pool_size, pool_stride=args.pool_stride,
                                 residual_blocks=args.residual_blocks, dilation_rate=args.dilation_rate,
                                 regularizer_term=args.regularizer_term)
        elif args.base_model == 'mlp':
            model = MLP(dense_dim=args.dense_dim, dense_dropout=args.dense_dropout, dense_layers=args.dense_layers)
        
        model.compile(loss=loss_fn, optimizer=opt, metrics=model_metrics)

        # ======================= Train model  =======================
        hist = model.fit(train_dataset, steps_per_epoch=train_steps,
                         epochs=args.epoch, validation_data=val_dataset,
                         callbacks=callbacks)
        model.summary()
        # ======================= Record training scores =======================
        train_acc = max(hist.history['binary_accuracy'])
        train_f1 = max(hist.history['f1_score'])
        train_roc_auc = max(hist.history['roc_auc'])
        train_pr_auc = max(hist.history['pr_auc'])
        train_mcc = max(hist.history['binary_mcc'])
        train_label01 = f'{sum(train_y == 0)},{sum(train_y == 1)}'
        val_acc = max(hist.history['val_binary_accuracy'])
        val_f1 = max(hist.history['val_f1_score'])
        val_roc_auc = max(hist.history['val_roc_auc'])
        val_pr_auc = max(hist.history['val_pr_auc'])
        val_mcc = max(hist.history['val_binary_mcc'])

        # ======================= Test model  =======================
        pred = model.predict(test_x)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        test_acc = accuracy_score(test_y, pred)
        test_f1 = f1_score(test_y, pred)
        test_mcc = matthews_corrcoef(test_y, pred)
        test_precision = precision_score(test_y, pred)
        test_recall = recall_score(test_y, pred)
        test_roc_auc = roc_auc_score(test_y, pred)

        # ======================= Record metrics  =======================
        run_args_values = [run_name, args.target, args.library, args.count_threshold, seed_entry, args.test_size,
                           args.val_size, args.rejection_sampling, args.minority_ratio, args.early_stop,
                           args.batch_size, args.epoch, args.learn_rate, args.kernel_size, args.filter_num,
                           args.stride, args.padding, args.pool_size, args.pool_stride, args.dense_dim,
                           args.dense_dropout, args.dense_layers, args.regularizer_term, args.base_model,
                           args.residual_blocks, args.dilation_rate]

        run_args_names = ['run_name', 'target', 'library', 'count_threshold', 'seed', 'test_size',
                          'val_size', 'rejection_sampling', 'minority_ratio', 'early_stop',
                          'batch_size', 'epoch', 'learn_rate', 'kernel_size', 'filter_num',
                          'stride', 'padding', 'pool_size', 'pool_stride', 'dense_dim',
                          'dense_dropout', 'dense_layers', 'regularizer_term', 'base_model',
                          'residual_blocks', 'dilation_rate']

        run_score_values = [train_acc, train_f1, train_roc_auc, train_pr_auc, train_mcc,
                            val_acc, val_f1, val_roc_auc, val_pr_auc, val_mcc,
                            test_acc, test_f1, test_mcc, test_precision, test_recall, test_roc_auc,
                            train_label01]

        run_score_names = ['train_acc', 'train_f1', 'train_roc_auc', 'train_pr_auc', 'train_binary_mcc',
                           'val_acc', 'val_f1', 'val_roc_auc', 'val_pr_auc', 'val_binary_mcc',
                           'test_acc', 'test_f1', 'test_mcc', 'test_precision', 'test_recall', 'test_roc_auc',
                           'train_label01']
        # combine lists
        full_scores = run_args_values + run_score_values
        full_names = run_args_names + run_score_names

        # ======================= Record metrics  =======================
        # record per run
        METRICS_DIR = f"{WORK_DIR}/metrics/{args.target}"
        if not os.path.exists(METRICS_DIR):
            os.makedirs(METRICS_DIR)

        metric_dict = dict(zip(full_names, full_scores))
        metric_df = info_df = pd.DataFrame.from_dict(metric_dict, orient='index')
        metric_df.rename(columns={0: timestr})
        metric_df.to_csv(f'{METRICS_DIR}/{run_name}_metrics.csv')

        # # ======================= Test on known VOC  =======================
        # create results folder
        VOC_DIR = f'{WORK_DIR}/voc/{args.target}'
        if not os.path.exists(VOC_DIR):
            os.makedirs(VOC_DIR)
        # load test variants
        test_var = pd.read_csv(f'{DATA_DIR}/VOC_Spike_all.csv', index_col=0)
        # encode depending on desired model
        var_x = encode_data(test_var.sequence, args.embedding)
        if args.base_model == 'mlp':
            var_x = var_x.reshape(var_x.shape[0], -1)
        test_predictions = model.predict(var_x)
        test_var[run_name] = test_predictions
        test_var.to_csv(f'{VOC_DIR}/{run_name}_test_variants.csv')
        # ======================= Save then Delete Model from Memory  =======================
        # check if models/ directory exists
        MODEL_DIR = f'{WORK_DIR}/models/{args.target}'
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        # only save models that have acceptable performance
        if test_mcc > 0.6:
            # save subclassed model by exporting weights (https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/tensorflow/saving_and_serializing.ipynb#scrollTo=wwT_YoKA0yQW)
            model.save_weights(f'{MODEL_DIR}/{run_name}', save_format='tf')

        # delete model to remove from memory
        del model
        keras.backend.clear_session()


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
