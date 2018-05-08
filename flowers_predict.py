"""
Distributed training procedure.

FLAGS.ckp_dir is the variable given by yarn -- "user.checkpoint.prefix" in yarn json configuration.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from inception.inception_predict import inference, parallel_inference, smart_inference

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    data_dir = '/Users/wanglifeng/code/deepstyle/data_dir/test'
    output_dir = '/Users/wanglifeng/code/deepstyle/data_dir/result'
    model_dir = '/Users/wanglifeng/code/deepstyle/data_dir/checkpoint'
    smart_inference(model_dir, data_dir, output_dir, 1, 0, True)
    print("End Predicting")
