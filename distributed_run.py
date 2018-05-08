"""
Distributed training procedure with porsche python_sdk.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os
import sys
currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath + os.sep + '../')
sys.path.append(currentPath + os.sep + '../..')

import time
import datetime
import traceback
import numpy as np

local_mode = True
if local_mode:
    from train.local_bootstrap import *
else:
    from flink_tensorflow.python_sdk.blink_bootstrap import *
    from flink_tensorflow.python_sdk.stream_io.stream_utils import *


import odps as Odps
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from inception import inception_distributed_train
from inception.flowers_data import FlowersData
from smart_io import odps_writer
from smart_io.console_log import log, log_error
from train.task_config import TaskConfig
from train.utils import parse_params, parent_directory, find_newest_subdir, \
        touch, tiny_odps_download, online_end_point, find_newest_partition
from train.inception_predict import extract_simhash
from flink_tensorflow.python_sdk.blink_bootstrap import *
from flink_tensorflow.python_sdk.stream_io.stream_utils import *
from flink_tensorflow.python_sdk.stream_io.feature_item import *


class Train(BlinkBootstrap):
    def run(self, target, data_stream, task_id, context):
        tf.logging.set_verbosity(tf.logging.INFO)
        param_map = parse_params(context.get_param())
        current_dir = os.path.dirname(os.path.abspath(__file__))

        log("task_id: %4d" % task_id)
        # Parse config parameters
        if param_map.has_key( "task_conf_file"):
            conf_file_path = os.path.join(parent_directory(current_dir, 2),
                                          param_map.get("task_conf_file"))
        else:
            conf_file_path = os.path.join(parent_directory(current_dir),
                                          'config/task_config.json')


        log("Will use task conf file %s" % conf_file_path)
        task_config = TaskConfig(param_map=param_map,
                                 conf_file_path=conf_file_path)
        bizdate = ((datetime.date.today() - datetime.timedelta(days=1))
                   .strftime('%Y%m%d'))
        task_config.add_if_not_contain("bizdate", bizdate)

        hash_codes_path = os.path.join(parent_directory(current_dir, 2),
                task_config.get_config("hash_codes_file"))
        log("load hash_codes_file: {}".format(hash_codes_path))
        hash_codes = np.load(hash_codes_path)

        hash_codes_num = int(task_config.get_config("hash_codes_num", "1"))
        if hash_codes_num < 1 or hash_codes_num > 6:
            log_error("hash_codes_num %d is invalid, use 1 instead."
                      % hash_codes_num)
            hash_codes_num = 1
        log("hash_codes_num: {}".format(hash_codes_num))

        batch_size = int(task_config.get_config("batch_size", '64'))
        log("batch_size: {}".format(batch_size))

        mode = task_config.get_config("mode", 'predict_once')
        if mode == 'predict_once':
            model_dir = task_config.get_config("pre_trained_model")
            data_dir = task_config.get_config("predict_input_dir")
            output_dir = task_config.get_config("predict_result_dir")
            flag_dir = os.path.join(task_config.get_config("predict_flag_dir"),
                                    task_config.bizdate)
            result_table = task_config.get_config("result_table")

            log("copy model files into local dir")
            local_model_dir = 'model_path'
            file_io.recursive_create_dir(local_model_dir)
            model_files = file_io.list_directory(model_dir)
            for model_file in model_files:
                log("copy %s" % model_file)
                file_io.copy(os.path.join(model_dir, model_file),
                             os.path.join(local_model_dir, model_file),
                             overwrite=True)

            # create flag_dir
            if task_id == 0:
                if not file_io.file_exists(flag_dir):
                    file_io.recursive_create_dir(flag_dir)
                odps_client = Odps.ODPS(task_config.odps_access_id,
                        task_config.odps_access_key, task_config.odps_project,
                        online_end_point)

                # delete this partitioin first.
                if result_table is not None:
                    odps_table = odps_client.get_table(result_table)
                    done_partition = "ds={}.done".format(task_config.bizdate)
                    odps_table.delete_partition(done_partition, if_exists=True)

            num_classes = ast.literal_eval(
                    task_config.get_config("num_classes", '1001'))

            # extract simhash features!
            with tf.device('/job:localhost/replica:0/task:0/cpu:0'):
                extract_simhash(
                        local_model_dir, data_dir, output_dir, batch_size,
                        context.get_worker_num(), task_id, hash_codes,
                        hash_codes_num, num_classes=num_classes)

            if result_table is not None:
                result_file = "simhash_result_%d" % task_id
                if file_io.file_exists(result_file):
                    odps_writer.upload(
                        access_id=task_config.odps_access_id,
                        access_key=task_config.odps_access_key,
                        task_id=task_id,
                        data_file=result_file,
                        project=task_config.odps_project,
                        table_name=result_table,
                        ds=task_config.bizdate,
                        split_key=";",
                        field_names=["content_id", "signature",
                                     "image_feature"],
                        field_types=["bigint", "string", "string"],
                        done_flag=False
                    )

                # after predict all images, touch done flag
                done_flag_filename = "{}.done".format(task_id)
                add_done_flag(flag_dir, done_flag_filename)
                log("add done file %s" % done_flag_filename)

                # sync and create done partition in chief worker.
                if (task_id == 0
                        and sync_by_hdfs(flag_dir, context.get_worker_num())):
                    odps_table.create_partition(done_partition,
                                                if_not_exists=True)
        else:
            log_error("Unknown mode %s" % mode)


def check_if_done(flag_dir, worker_num, task_id, output_dir,
                  result_done_filename):
    finally_done_flag = os.path.join(output_dir, result_done_filename)
    if task_id == 0:
        done_count = len(file_io.list_directory(flag_dir))
        while done_count < worker_num:
            log("{} workers not done yet, sleep 30s ...".format(
                worker_num - done_count))
            time.sleep(30)
            done_count = len(file_io.list_directory(flag_dir))
        else:
            touch(result_done_filename)
            file_io.copy(result_done_filename, finally_done_flag,
                         overwrite=True)
    else:
        while not file_io.file_exists(finally_done_flag):
            log("{} has not been touched, will sleep 30s".format(
                finally_done_flag))
            time.sleep(30)


def sync_by_hdfs(flag_dir, worker_num, msg="sync_by_hdfs"):
    done_count = count_done_flag(flag_dir)
    try_count = 120
    while done_count < worker_num and try_count > 0:
        log("[{}] {} workers not done yet, sleep 30s, try_count {} ...".format(
            msg, worker_num - done_count, try_count))
        time.sleep(30)
        try_count -= 1
        done_count = count_done_flag(flag_dir)

    return done_count == worker_num


def count_done_flag(flag_dir):
    done_count = len([x for x in file_io.list_directory(flag_dir)
                      if '.done' in x])
    return done_count


def add_done_flag(flag_dir, done_flag_filename):
    touch(done_flag_filename)
    file_io.copy(done_flag_filename, os.path.join(flag_dir, done_flag_filename),
                 overwrite=True)
    os.remove(done_flag_filename)


def serialize_id(raw_id, serialized_map):
    if raw_id in serialized_map:
        serialized_id = serialized_map[raw_id]
    else:
        serialized_id = len(serialized_map)
    return serialized_id


def main(unused_argv):
    Train().start(unused_args=unused_argv)


if __name__ == '__main__':
    if local_mode:
        currentPath = os.path.split(os.path.realpath(__file__))[0]
        ckp_dir = currentPath + os.sep + 'data_dir/checkpoint'
        upstream_schema = None
        data_file = currentPath + os.sep + 'data_dir/test'
        Train().start(ckp_dir, upstream_schema, data_file)
    else:
        Train().start()
