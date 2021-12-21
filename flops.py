# Evaluation of the flops required to calculate the model
import os
import shutil
import sys
import argparse


# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,LearningRateScheduler

from dataset_utils import input_fn_trn, input_fn_test
from customcnn import customcnn


import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
print('TensorFlow:', tf.__version__)

#model = tf.keras.applications.ResNet50()
model = customcnn(input_shape=(125, 100, 4),classes=2,filters=[8,16,32,64,128])

forward_pass = tf.function(
    model.call,
    input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

graph_info = profile(forward_pass.get_concrete_function().graph,
                        options=ProfileOptionBuilder.float_operation())

# The //2 is necessary since `profile` counts multiply and accumulate
# as two flops, here we report the total number of multiply accumulate ops
#flops = graph_info.total_float_ops // 2
flops = graph_info.total_float_ops
print('Flops: {:,}'.format(flops))


