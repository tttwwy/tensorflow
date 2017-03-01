# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
                            'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 1, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            # global_step = tf.Variable(0, name='global_step', trainable=False)
            global_step = tf.contrib.framework.get_or_create_global_step()

            input = tf.placeholder("float")
            label = tf.placeholder("float")

            weight = tf.get_variable("weight", [1], tf.float32, initializer=tf.random_normal_initializer())
            biase = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer())
            pred = tf.multiply(input, weight) + biase

            loss_value = loss(label, pred)
            is_chief = (FLAGS.task_index == 0)
            opt = tf.train.AdamOptimizer()
            if issync == 1:
                opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=len(worker_hosts),
                                               total_num_replicas=len(worker_hosts))
            train_op = opt.minimize(loss_value,global_step=global_step)
            init_op = tf.global_variables_initializer()
            hooks = [tf.train.StopAtStepHook(last_step=1000000)]
            if issync == 1:
                hooks.append(opt.make_session_run_hook(is_chief))
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=is_chief,
                                                    checkpoint_dir="./checkpoint/",
                                                    hooks=hooks,
                                                   config=config) as sess:
                if is_chief and issync == 1:
                    sess.run(init_op)
                while not sess.should_stop():
                    train_x = np.random.randn(1)
                    train_y = 2 * train_x + np.random.randn(1) * 0.33 + 10
                    _, loss_v, step = sess.run([train_op, loss_value, global_step],
                                               feed_dict={input: train_x, label: train_y})
                    if step % steps_to_validate == 0:
                        w, b = sess.run([weight, biase])
                        print("step: {0}, weight: {1}, biase: {2}, loss: {3} task id:{4}".format(step,w,b,loss_v,FLAGS.task_index),file=sys.stderr)
                        # print("step: %d, weight: %f, biase: %f, loss: %f" % (step, w, b, loss_v),file=sys.stderr)




def loss(label, pred):
    return tf.square(label - pred)


if __name__ == "__main__":
    tf.app.run()
