#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <0ahttp://www.gnu.org/licenses/>.
#

from dataloader import DataGenerator
from model import unet
import datetime
from argparser import args
import os
from IPython import embed

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)

# If hyperthreading is enabled, then use
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow.compat.v1 as tf


if args.keras_api:
    import keras as K
else:
    from tensorflow.compat.v1 import keras as K #excuted


import pickle, gzip, pickletools
import tensorflow as tf2

tf2.debugging.set_log_device_placement(False)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

strategy = tf2.distribute.MirroredStrategy()
NUM_GPU = strategy.num_replicas_in_sync
print("Number of devices: {}".format(NUM_GPU))

if args.varification_test:
	args.epochs = 0
else:
	args.epochs = 500
args.lr = 0.1*args.lr # the original lr was 0.01
args.bz = 1*NUM_GPU 

args.patch_height = 144
args.patch_width = 144
args.patch_depth = 144

args.print_model = True
print("Args = {}".format(args))

CHANNELS_LAST = True

if CHANNELS_LAST:
   print("Data format = channels_last")
else:
   print("Data format = channels_first")

# os.system("lscpu")
start_time = datetime.datetime.now()
print("Started script on {}".format(start_time))

#os.system("uname -a")
print("TensorFlow version: {}".format(tf.__version__))

print("Keras API version: {}".format(K.__version__))

# Optimize CPU threads for TensorFlow
CONFIG = tf.compat.v1.ConfigProto(
    inter_op_parallelism_threads=args.interop_threads,
    intra_op_parallelism_threads=args.intraop_threads)

SESS = tf.compat.v1.Session(config=CONFIG)

K.backend.set_session(SESS)


with strategy.scope():
    unet_model = unet(use_upsampling=args.use_upsampling, #False
                      learning_rate=args.lr,
                      n_cl_in=args.number_input_channels,
                      n_cl_out=3,  # single channel (greyscale)
                      feature_maps = args.featuremaps,
                      dropout=0.2,
                      print_summary=args.print_model,
                      channels_last = CHANNELS_LAST, # channels first or last
                      batch_size = args.bz,
                      num_gpu= NUM_GPU,
                      height = args.patch_height,
                      width = args.patch_width,
                      depth = args.patch_depth,
                      )

    unet_model.model.compile(optimizer=unet_model.optimizer,
                  loss=unet_model.loss,
                  metrics=unet_model.metrics,
                  run_eagerly = True)

    # Save best model to hdf5 file
    saved_model_directory = os.path.dirname(args.saved_model)
    try:
        os.stat(saved_model_directory)
    except:
        os.mkdir(saved_model_directory)

    # If there is a current saved file, then load weights and start from
    # there.
    if os.path.isfile(args.saved_model):
        unet_model.model.load_weights(args.saved_model)

    checkpoint = K.callbacks.ModelCheckpoint(args.saved_model,
                                           verbose = 1,
                                           save_best_only = True)

# TensorBoard
currentDT = datetime.datetime.now()
tb_logs = K.callbacks.TensorBoard(log_dir=os.path.join(
    saved_model_directory, "tensorboard_logs", currentDT.strftime("%Y/%m/%d-%H:%M:%S")), update_freq="batch")

# Keep reducing learning rate if we get to plateau
reduce_lr = K.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=1, min_delta=1e-5, min_lr=1e-20) # min_lr was 0.0001

callbacks = [checkpoint, tb_logs, reduce_lr]

training_data_params = {"dim": ( args.patch_height, args.patch_width, args.patch_depth),
                        "batch_size": args.bz,
                        "n_in_channels": args.number_input_channels,
                        "n_out_channels": 3,
                        "shuffle": True,
                        "seed": args.random_seed,
                        "varification_test": args.varification_test}

training_generator = DataGenerator("train", args.data_path,
                                   **training_data_params)
training_generator.print_info()

validation_data_params = {"dim": (args.patch_height, args.patch_width, args.patch_depth),
                          "batch_size": args.bz,
                          "n_in_channels": args.number_input_channels,
                          "n_out_channels": 3,
                          "shuffle": False,
                          "seed": args.random_seed,
                          "varification_test": args.varification_test}
validation_generator = DataGenerator("validate", args.data_path,
                                     **validation_data_params)
validation_generator.print_info()


unet_model.model.fit_generator(training_generator,
                    epochs=args.epochs, verbose=1,
                    validation_data=validation_generator,
                    callbacks=callbacks,
                    max_queue_size=args.num_prefetched_batches,
                    workers=args.num_data_loaders,
                    use_multiprocessing=False)


test_data_params = {"dim": (args.patch_height, args.patch_width, args.patch_depth),
                          "batch_size": 1,
                          "n_in_channels": args.number_input_channels,
                          "n_out_channels": 3,
                          "shuffle": False,
                          "seed": args.random_seed,
                          "varification_test": args.varification_test}

# Evaluate final model on test holdout set
testing_generator = DataGenerator("test", args.data_path,
                                     **test_data_params)
testing_generator.print_info()

# Load the best model
print("Loading the best model: {}".format(args.saved_model))
unet_model.model.load_weights(args.saved_model)

save_inps = 1
save_targs = 1
save_preds = 1
compute_scores = 1
calculate_target_max_abs = True

if save_inps or save_targs:
    print('Loading test inputs and targets...')
    test_inputs, test_targets = testing_generator.get_test_inputs_targets()

if calculate_target_max_abs:
    print('Computing max ABSOLUTE value in test target set:')
    test_targ_max = testing_generator.get_test_targets_max_abs()
    print('\ntest_targ_max = ', test_targ_max)

if save_preds:
    print('Computing predictions of test set:')
    preds = unet_model.model.predict_generator(testing_generator, verbose=1)  # preds.shape[38,144,144,144,1]

    print('Saving predictions to test_preds.pkl.')
    with gzip.open('./saved_data/test_preds.pkl', 'wb') as f:
        pickled = pickle.dumps(preds, protocol=4)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)

if save_inps:
    print('Saving test_inputs.')
    with gzip.open(os.path.join('./saved_data/', 'test_inputs.pkl'), 'wb') as f:
        pickled = pickle.dumps(test_inputs, protocol=4)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)
        f.write(optimized_pickle)

if save_targs:
    print('Saving test_targets.')
    with gzip.open(os.path.join('./saved_data/', 'test_targets.pkl'), 'wb') as f:
        pickled = pickle.dumps(test_targets, protocol=4)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)

if compute_scores:
    print('Computing scores for test set:')
    scores = unet_model.model.evaluate_generator(testing_generator, verbose=1)
    print("Final model metrics on test dataset:")
    for idx, name in enumerate(unet_model.model.metrics_names):
        print("{} \t= {}".format(name, scores[idx]))


stop_time = datetime.datetime.now()
print("Started script on {}".format(start_time))
print("Stopped script on {}".format(stop_time))
print("\nTotal time for training model = {}".format(stop_time - start_time))
