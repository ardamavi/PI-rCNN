"""
######################################################
# Training Procedure
# by Arda Mavi
# The University of Arizona
# Department of Aerospace and Mechanical Engineering
######################################################

#### Module: "training_procedure.py"

#### Description: Train model.

#### Functions:
#---------------------------------------------------------

Func: training(Model, inital_data, LossFunc, batch_size=100, dataset_time_steps=1000,
              epoch=2000, learning_rate=0.001, checkpoint_dir="Checkpoints",
              model_folder="Model", loss_file="training_loss.npy", append_loss=False)

Args:
    - Model (Tensorflow Object):
        Tensorflow Model,
        can be created using Model Procedure module.
    - initial_data (List):
        List of initial condition and initial states (h and c) of ConvLSTM.
    - LossFunc (Tensorflow Object):
        Tensorflow custom loss function,
        can be created using Model Procedure module.
    - batch_size (Int):
        Default: 100
        Number of time steps (considering dt) to calculate loss and gradients.
    - dataset_time_steps (Int):
        Default: 1000
        Total time step size (considering dt) that will be used in training.
    - epoch (Int):
        Default: 2000
        Number of epoch for training.
    - learning_rate (Float):
        Default: 0.001
        Initial learning rate of the training cycle.
    - checkpoint_dir (String):
        Default: "Checkpoints"
        Direction of Model checkpoint and Loss records during the training.
    - model_folder (String):
        Default: "Model"
        Direction to save trained Model weights.
        Model weights will be saved to <checkpoint_dir>+"/"+<model_folder>
    - loss_file (String):
        Default: "training_loss.npy"
        File name for saving loss changes during the training.
        The function will save loss records into a Numpy file, with ".npy" file extension.
        Loss records will be saved to <checkpoint_dir>+"/"+<loss_file>+".npy"
    - append_loss (Boolean):
        Default: False
        If True, append new loss record as continues of previous saved record.
        ! Previous loss record file must be exist and
          it's name must have the same name with <loss_file>.

Returns:
    - Model (Tensorflow Object):
        Trained Tensorflow Model.

#---------------------------------------------------------
"""

import os
import sys
import random
import scipy.io
import numpy as np
from time import time
import tensorflow as tf
from tqdm import tqdm as ProgressBar
from tensorflow.keras import optimizers as K_Optim

# Import Model Procedure:
sys.path.append("../Model")
from model_procedure import get_inital_states


# Set TensorFlow Backend for Channel Last Data Type:
tf.keras.backend.set_image_data_format('channels_last')
# Prevent TennSorflow notes during training:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# Model Training:
def training(Model, initial_data, LossFuncs, batch_size=100, dataset_time_steps=1000,
            epoch=2000, learning_rate=0.001, checkpoint_dir="Checkpoints",
            model_folder="Model", loss_file="training_loss.npy", append_loss=False):

    # Create checkpoints dir, if not exists:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Recod loss:
    loss_record = []

    # Loss Functions:
    LossFunc, Latent_LossFunc = LossFuncs

    # Specifying batch size for loss calculation:
    num_batch = int(dataset_time_steps/batch_size)

    # Exponential learning rate decay:
    # learning_rate = K_Optim.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=1000, decay_rate=0.96)

    # Get Optimizer:
    Optimizer = K_Optim.RMSprop(learning_rate=learning_rate)

    # Epoch:
    for epoch_index in range(epoch):

        # Print epoch:
        print("Epoch: {0}/{1}".format(epoch_index+1, epoch))

        # Progress Bar:
        progress_bar = ProgressBar( total = num_batch * batch_size )

        # Initial conidtion and initial states:
        prev_out, prev_h, prev_c = initial_data

        # Calculate losses with smaller time steps:
        for _ in range(num_batch):
            # Gradient Session:
            with tf.GradientTape() as tape:
                # Output list:
                outputs = tf.reshape(tf.constant([], dtype='float32'), (0,)+prev_out.shape[1:])
                encode_list = tf.reshape(tf.constant([], dtype='float32'), (0,)+prev_h.shape[1:])

                # Time Steps:
                for _ in range(batch_size):
                    prev_out, prev_h, prev_c, encode  = Model([prev_out, prev_h, prev_c], training=True)
                    outputs = tf.concat([outputs, prev_out], axis=0)
                    encode_list = tf.concat([encode_list, encode], axis=0)

                    # Progress Bar update:
                    progress_bar.update(1)

                # Calculate loss along the batches:
                loss_main = LossFunc(outputs)
                loss_latent = Latent_LossFunc(encode_list)
                loss = (loss_main*0.6) + (loss_latent*0.4)

            # Calculate gradients:
            gradients = tape.gradient( loss, Model.trainable_weights )

            # Gradient clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, 1.)

            # Optimization:
            Optimizer.apply_gradients(zip( gradients, Model.trainable_weights ))

        # Progress bar complated:
        progress_bar.close()

        # Record loss values:
        loss_record.append(loss.numpy())

        # Print loss of the epoch:
        print("Loss: {0}\n".format(loss_record[-1]))

        # Quit, if loss is nan:
        if np.isnan(loss_record[-1]):
            raise RuntimeError('"Nan" loss during training!')

        # Save Model checkpoint if loss is better:
        if min(loss_record) == loss_record[-1]:
            Model.save_weights(checkpoint_dir+"/"+model_folder+"/")

        # Flush output buffer:
        sys.stdout.flush()

    # Save training loss, append to the old record, if append_loss=True:
    if append_loss:
        try:
            loss_record = list(np.load(checkpoint_dir+"/"+loss_file)) + loss_record[1:]
        except:
            pass
    np.save(checkpoint_dir+"/"+loss_file, loss_record)

    # Return trained model:
    return Model
