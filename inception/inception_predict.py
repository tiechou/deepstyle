# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import math
import os.path
import thread
import traceback
import numpy as np
import tensorflow as tf
import multiprocessing
import time
from datetime import datetime
from tensorflow.python.lib.io import file_io
from inception.build_image_data import ImageCoder, _process_image
from inception.image_processing import image_preprocessing

FLAGS = tf.app.flags.FLAGS


def create_graph(model_dir):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    print("build graph from ckpt done!")


def _split(data_list, part_num):
    slice_size_floor = len(data_list) // part_num
    slice_size_ceil = len(data_list) // part_num + 1
    residual = len(data_list) % part_num
    return [data_list[i:i + slice_size_ceil] for i in xrange(residual * slice_size_ceil) if
            i % slice_size_ceil == 0] + [data_list[i:i + slice_size_floor] for i in
                                         xrange(residual * slice_size_ceil, len(data_list)) if
                                         i % slice_size_floor == 0]


def _get_file_list(data_dir, worker_num, task_id):
    all_files = [os.path.join(data_dir, file_name) for file_name in file_io.list_directory(data_dir)]
    input_file_list = _split(all_files, worker_num)
    if len(input_file_list) <= task_id:
        print("Worker %d will process nothing!" % task_id)
        return []
    else:
        return input_file_list[task_id]


def parallel_inference(model_dir, data_dir, output_dir, worker_num, task_id, process_num):
    input_files = _get_file_list(data_dir, worker_num, task_id)
    print("input files %s " % input_files)
    assert len(input_files) >= process_num, "Too less data for parallel %s" % process_num
    # Creates graph from saved GraphDef.
    create_graph(model_dir)
    np.set_printoptions(threshold=np.nan, linewidth=np.nan, precision=17, suppress=True)
    local_result_file_name = "inception_result_%d" % task_id
    local_result_file = file_io.FileIO(local_result_file_name, 'a+')
    pool = multiprocessing.Pool(processes=process_num)
    for file_names in _split(input_files, process_num):
        pool.apply_async(inference_on_images, (local_result_file, file_names))
    pool.close()
    pool.join()
    local_result_file.close()
    put_files_to_hdfs([local_result_file_name], output_dir)


def inference(model_dir, data_dir, output_dir, worker_num, task_id):
    input_files = _get_file_list(data_dir, worker_num, task_id)
    if not input_files:
        return
    else:
        print("input files %s " % input_files)
    # Creates graph from saved GraphDef.
    create_graph(model_dir)
    np.set_printoptions(threshold=np.nan, linewidth=np.nan, precision=17, suppress=True)
    local_result_file_name = "inception_result_%d" % task_id
    local_result_file = file_io.FileIO(local_result_file_name, 'a+')
    inference_on_images(local_result_file, input_files)
    local_result_file.close()
    put_files_to_hdfs([local_result_file_name], output_dir)


def inference_on_images(output_file, image_list):
    print("[%s] inference on totally %d images" % (thread.get_ident(), len(image_list)))
    for index, image_file_path in enumerate(image_list):
        file_name = image_file_path.split('/')[-1][:-4]
        print("[%d] inference on image %s" % (index, file_name))
        try:
            with file_io.FileIO(image_file_path, 'r') as image_file_path:
                image_data = image_file_path.read()
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                    # Some useful tensors:
                    # 'softmax:0': A tensor containing the normalized prediction across
                    #   1000 labels.
                    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
                    #   float description of the image.
                    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
                    #   encoding of the image.
                    # Runs the softmax tensor by feeding the image_data as input to the graph.
                    softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
                    predictions = sess.run(softmax_tensor,
                                           {'DecodeJpeg/contents:0': image_data})
                    predictions = np.squeeze(predictions).tolist()
                    item_id, category = file_name.split('_')
                    output_file.write('{};{};{};{}\n'.format(item_id, "", predictions, category))
        except:
            print("[inference on image {} error]: {}".format(file_name, traceback.format_exc()))


def put_files_to_hdfs(file_list, dest_dir):
    for file_path in file_list:
        file_io.copy(file_path, os.path.join(dest_dir, file_path), overwrite=True)


def smart_inference(checkpoint_dir, data_dir, output_dir, worker_num, task_id, output_class=False):
    from inception import inception_model
    input_files = _get_file_list(data_dir, worker_num, task_id)
    if not input_files:
        return
    else:
        print("input files %s " % input_files)
    with tf.Graph().as_default():
        images, item_ids, categories = process_images(input_files)
        np.set_printoptions(threshold=np.nan, linewidth=np.nan, precision=17, suppress=True)
        local_result_file_name = "inception_result_%d" % task_id
        local_result_file = file_io.FileIO(local_result_file_name, 'w')
        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = 5 + 1

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, _, pool2048 = inception_model.inference(images, num_classes)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            inception_model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/imagenet_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Successfully loaded model from %s at step=%s.' %
                      (ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found')
                return
            try:
                if output_class:
                    result_classes = sess.run(tf.arg_max(logits, 1)).tolist()
                    result_vectors = sess.run(pool2048).tolist()
                    for item_id, result_vector, category, result_class in zip(item_ids, result_vectors, categories,
                                                                              result_classes):
                        local_result_file.write('{};{};{};{}\n'.format(item_id, result_class, result_vector, category))
                else:
                    predictions = sess.run(pool2048)
                    # predictions = np.squeeze(predictions).tolist()
                    predictions = predictions.tolist()
                    for item_id, category, prediction in zip(item_ids, categories, predictions):
                        local_result_file.write('{};{};{};{}\n'.format(item_id, "", prediction, category))
                local_result_file.flush()
                local_result_file.close()
                put_files_to_hdfs([local_result_file_name], output_dir)
            except Exception as e:
                print(traceback.format_exc())


def process_images(input_files):
    coder = ImageCoder()
    image_buffer_list = []
    item_id_list = []
    category_list = []
    for filename in input_files:
        try:
            image_buffer, _, _ = _process_image(filename, coder)
            item_id, category = filename.split('/')[-1].split('.')[0].split('_')
            image_buffer = image_preprocessing(image_buffer, "", False)
            image_buffer_list.append(image_buffer)
            item_id_list.append(item_id)
            category_list.append(category)
        except:
            print("[Process image {} error]: {}".format(filename, traceback.format_exc()))
    height = FLAGS.image_size
    width = FLAGS.image_size
    depth = 3
    batch_size = len(image_buffer_list)
    images = tf.stack(image_buffer_list)
    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, height, width, depth])
    return images, item_id_list, category_list


def latest_meta(latest_model_path):
    pattern = re.compile(r'(.*/model.ckpt-\d+)')
    match = pattern.match(latest_model_path)
    if match:
        return '{}.meta'.format(match.group())
    else:
        raise ValueError('No corresponding meta file of model path {}'.format(latest_model_path))
