#!encoding=utf-8
"""

"""
import os
import sys
import math
import time
import json
import random
import hashlib
import datetime
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from layer_util import *
from data_util import read_data
from hyper_param import param_dict as pd
from gru import GRU
from replay_buffer import RB
import resource

###### parameters for distributed training ######
flags = tf.app.flags
flags.DEFINE_string("model_path", "train_model", "Saved model dir.")
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs.")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs.")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = flags.FLAGS

g_batch_counter = 0
g_working_mode = None
g_training = None

g_v_loss_sum = 0.
g_v_loss_cnt = 0
g_p_loss_sum = 0.
g_p_loss_cnt = 0
g_click_loss_sum = 0.
g_click_loss_cnt = 0
g_dwell_loss_sum = 0.
g_dwell_loss_cnt = 0

g_rb = None
pd["num_objs"] = 2#多少个参数
pd["grad_clip"] = 5
pd["uid_size"] = 2*12
pd['emb_dim'] = 128


def sum_dense(x, y, w1, w2):
    return tf.scalar_mul(w1, x) + tf.scalar_mul(w2, y)


#这个函数是actor函数
class PolicyNetwork(object):
    def __init__(self, global_step):
        self.global_step = global_step
        # 用户的id作为一个占位符，调用函数的时候会传入值到这个placeholder里
        self.sph_user = tf.placeholder(tf.int32, name='sph_user')
        # 目标的参数
        self.sph_weights = tf.placeholder(tf.float32, name='sph_weights')
        # 策略梯度作为placeholder传入
        self.a_grads = tf.placeholder(tf.float32)

        #这里使用了两个网络，main和target，只更新main，之后会吧main的参数复制给target
        # main networks
        self.dst_weight, self.mpa, self.mea = self.create_graph('main')#self.mpa
        # target networks
        _, self.tpa, self.tea = self.create_graph('target')

        # 计算loss
        self.loss = tf.losses.mean_squared_error(self.dst_weight, self.mea)

        # 将需要更新的参数传入到params里
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main/p')
        params.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main/feat_embedding'))

        # 这里吧梯度做一个截断，防止梯度过大或者过小
        self.grads = tf.clip_by_global_norm(tf.gradients(self.loss, params), pd['grad_clip'])[0]
        policy_grads = \
        tf.clip_by_global_norm(tf.gradients(ys=self.mea, xs=params, grad_ys=self.a_grads), pd['grad_clip'])[0]
        opt1 = tf.train.AdamOptimizer(-pd['lr_pg'])
        opt2 = tf.train.AdamOptimizer(pd['lr_sl'])

        with tf.variable_scope("train-p"):
            self.opt_a1 = opt1.apply_gradients(zip(policy_grads, params), global_step=self.global_step)
            self.opt_a2 = opt2.apply_gradients(zip(self.grads, params), global_step=self.global_step)

        # 这里设计一个算子sync_op，可以讲main网络的参数复制给target，只是这里是soft copy
        self.m_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="main/p")
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target/p")
        alpha = pd['double_networks_sync_step']
        self.sync_op = [tf.assign(t, (1.0 - alpha) * t + alpha * m) for t, m in zip(self.t_params, self.m_params)]


    def create_graph(self, scope):
        global g_training
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # 定义embedding矩阵（推荐里的，FL不用）
            feat_dict = get_embeddings(pd['uid_size'],
                                       pd['emb_dim'],
                                       scope='feat_embedding',
                                       zero_pad=False,
                                       partitioner=tf.fixed_size_partitioner(pd['embed_dict_part'], 0))
            with tf.variable_scope('p'):
                n_batch = pd['batch_size']
                embed_dim = pd['emb_dim']
                num_objs = pd["num_objs"]

                #user的embedding
                user_embed = tf.nn.embedding_lookup(feat_dict, self.sph_user, sp_weights=None,
                                                           partition_strategy='div', combiner='mean')
                step_s = tf.reshape(user_embed, shape=[n_batch, embed_dim])

                sph_weights = tf.reshape(self.sph_weights, shape=[pd["batch_size"], pd["num_objs"]])

                #得到action---也就是weight
                p_action = tf.layers.dense(step_s, num_objs)#mlp（输入，输出client个数）

                #ddpg里，action需要做扩展，所以这里在原始的action上加一些噪音
                explore_action = tf.truncated_normal(shape=[pd['batch_size'], pd['num_objs']],#噪音避免过拟合，可加可不加
                                                     mean=0,
                                                     stddev=pd['action_explore'],
                                                     dtype=tf.float32)
                # action with exploration
                e_action = p_action + explore_action

                # softmax
                p_action = tf.nn.softmax(p_action, dim=-1, name="p_action_softmax")#权重合为1
                e_action = tf.nn.softmax(e_action, dim=-1, name="e_action_softmax")
                print
                'softmax p_action.shape:', p_action.shape
                print
                'softmax doc_embed.shape:', doc_embed.shape

            return sph_weights, p_action, e_action#p_action是actor产生的权重

    # 得到p_action
    def act_for_train(self, sess, ph_dict):
        return sess.run(self.mpa, feed_dict={
            self.sph_user: ph_dict['user']}) #FL中所有client传的参数

    # 得到p_action
    def act_for_eval(self, sess, ph_dict):
        return sess.run(self.mpa, feed_dict={
            self.sph_user: ph_dict['user']})

    #使用learn学习参数
    def learn(self, sess, ph_dict):
        loss, _, _ = sess.run([self.loss, self.opt_a1, self.opt_a2], feed_dict={self.a_grads: ph_dict['a_grads'],
                                                                                self.sph_user: ph_dict['user'],
                                                                                self.sph_weights: ph_dict['weight'],
                                                                                self.sph_doc: ph_dict['doc'],
                                                                                self.sph_con: ph_dict['con']})
        global g_p_loss_sum, g_p_loss_cnt
        g_p_loss_sum += loss
        g_p_loss_cnt += 1


#这个函数是critic函数
class ValueNetwork(object):#actor选出weight后，得到global model，下放给client，继续训练，有loss，
    #就是reward。critic是在预测这个loss多大，越大的话weight越差。client还要传loss除了参数外
    #传的loss是第一次global model和现实值的误差，不是本地迭代多轮后的
    def __init__(self, global_step):
        self.global_step = global_step
        self.sph_user = tf.placeholder(tf.int32, name='sph_user')
        # weights of objs[w1,w2]
        self.sph_weights = tf.placeholder(tf.float32, name='sph_weights')
        self.ph_reward = tf.placeholder(tf.float32, shape=[pd['batch_size']], name='ph_reward')
        self.ph_nq = tf.placeholder(tf.float32, shape=[pd['batch_size'], 1], name='ph_nq')

        # main networks
        self.dst_embed, self.mq = self.create_graph('main')
        # target networks
        _, self.tq = self.create_graph('target')

        diff = tf.reshape(self.ph_reward, [-1]) + tf.scalar_mul(tf.constant(pd['rl_gamma']),
                                                                tf.reshape(self.ph_nq, [-1])) - tf.reshape(self.mq,
                                                                                                           [-1])
        self.loss = tf.reduce_mean(tf.square(diff))

        #
        self.a_grads = tf.clip_by_global_norm(tf.gradients(self.mq, self.dst_embed), pd['grad_clip'])[0]

        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main/q')
        vs.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main/feat_embedding'))
        self.grads = tf.clip_by_global_norm(tf.gradients(self.loss, vs), pd['grad_clip'])[0]
        with tf.variable_scope('opt_q'):
            optimizer = tf.train.AdamOptimizer(pd['lr_q'])
            self.opt = optimizer.apply_gradients(zip(self.grads, vs), global_step=global_step)

        self.m_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="main/q")
        self.m_params.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main/feat_embedding'))
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target/q")
        self.t_params.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target/feat_embedding'))

        alpha = pd['double_networks_sync_step']
        self.sync_op = [tf.assign(t, (1.0 - alpha) * t + alpha * m) for t, m in zip(self.t_params, self.m_params)]


    def create_graph(self, scope):
        global g_training
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            feat_dict = get_embeddings(pd['uid_size'],
                                       pd['emb_dim'],
                                       scope='feat_embedding',
                                       zero_pad=False,
                                       partitioner=tf.fixed_size_partitioner(pd['embed_dict_part'], 0))
            with tf.variable_scope('q'):
                n_batch = pd['batch_size']
                embed_dim = pd['emb_dim']
                num_objs = pd["num_objs"]


                user_embed = tf.nn.embedding_lookup(feat_dict, self.sph_user, sp_weights=None,
                                                           partition_strategy='div', combiner='mean')
                user_field = tf.reshape(user_embed, shape=[n_batch, embed_dim])

                # weights
                weight_field = tf.reshape(self.sph_weights, shape=[n_batch, num_objs])

                step_a = tf.reshape(user_field, shape=[n_batch, -1])

                step_w = tf.identity(tf.reshape(weight_field, shape=[n_batch, -1]), name="weight_embed")

                step_s = tf.identity(tf.concat([step_a, step_w], axis=1), name='gru_x')

                q = tf.concat(step_s, axis=1)
        return weight_field, q

    # call for evaluating policy networks
    def critic(self, sess, ph_dict):
        return sess.run(self.mq, feed_dict={self.dst_embed: ph_dict['action'],
                                            self.sph_user: ph_dict['user']})

    # call for temporal-diffenence learning
    def target_q(self, sess, ph_dict):
        return sess.run(self.tq, feed_dict={
            self.sph_user: ph_dict['user'],
            self.sph_doc: ph_dict['doc'],
            self.sph_weights: ph_dict['weight'],
            self.sph_con: ph_dict['con']})

    # call for evaluating value networks
    def main_q(self, sess, ph_dict):
        return sess.run(self.mq, feed_dict={
            self.sph_user: ph_dict['user'],
            self.sph_doc: ph_dict['doc'],
            self.sph_weights: ph_dict['weight'],
            self.sph_con: ph_dict['con']})

    # call for learning from data
    def learn(self, sess, ph_dict):
        loss, _ = sess.run([self.loss, self.opt], feed_dict={self.ph_nq: ph_dict['next_q'],
                                                             self.ph_reward: ph_dict['grad_reward'],
                                                             self.sph_user: ph_dict['user'],
                                                             self.sph_weights: ph_dict['weight']})
        global g_v_loss_sum, g_v_loss_cnt
        g_v_loss_sum += loss
        g_v_loss_cnt += 1

    # call for training policy networks
    def pg(self, sess, ph_dict):
        return sess.run(self.a_grads, feed_dict={self.dst_embed: ph_dict['action'],
                                                 self.sph_user: ph_dict['user'],
                                                 self.sph_clk_seq: ph_dict['clk_seq']})


# objective related model
class ObjNetwork(object):
    def __init__(self, global_step):
        self.global_step = global_step
        # placeholder
        self.sph_user = tf.placeholder(tf.int32, name='sph_user')
        self.ph_click_reward = tf.placeholder(tf.float32, shape=[pd['batch_size']],
                                              name='ph_click_reward')
        self.ph_dwell_reward = tf.placeholder(tf.float32, shape=[pd['batch_size']],
                                              name='ph_dwell_reward')
        self.ph_click_nq = tf.placeholder(tf.float32, shape=[pd['batch_size'], 1], name='ph_click_nq')
        self.ph_dwell_nq = tf.placeholder(tf.float32, shape=[pd['batch_size'], 1], name='ph_dwell_nq')
        self.ph_click_weight = tf.placeholder(tf.float32, shape=[pd['batch_size']],
                                              name='ph_click_weight')
        self.ph_dwell_weight = tf.placeholder(tf.float32, shape=[pd['batch_size']],
                                              name='ph_dwell_weight')
        self.ph_loss_pos = tf.placeholder(tf.int32, shape=[1], name='ph_loss_pos')
        # self.ph_loss_pos = tf.placeholder(tf.int32, shape=[5], name='ph_loss_pos')

        # click main networks
        _, self.click_mq = self.create_graph('main_click')
        # dwell target networks
        _, self.click_tq = self.create_graph('target_click')
        # dwell main networks
        _, self.dwell_mq = self.create_graph('main_dwell')
        # dwell target networks
        _, self.dwell_tq = self.create_graph('target_dwell')

        click_diff = tf.reshape(self.ph_click_reward, [-1]) + tf.scalar_mul(tf.constant(pd['rl_gamma']),
                                                                            tf.reshape(self.ph_click_nq,
                                                                                       [-1])) - tf.reshape(
            self.click_mq, [-1])
        self.click_loss = tf.reduce_mean(tf.square(click_diff), name="click_loss")
        dwell_diff = tf.reshape(self.ph_dwell_reward, [-1]) + tf.scalar_mul(tf.constant(pd['rl_gamma']),
                                                                            tf.reshape(self.ph_dwell_nq,
                                                                                       [-1])) - tf.reshape(
            self.dwell_mq, [-1])
        self.dwell_loss = tf.reduce_mean(tf.square(dwell_diff), name="dwell_loss")

        # click obj & dwell obj
        click_vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_click/q')
        click_vs.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_click/feat_embedding'))
        dwell_vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_dwell/q')
        dwell_vs.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_dwell/feat_embedding'))
        self.click_grads = tf.clip_by_global_norm(tf.gradients(self.click_loss, click_vs), pd['grad_clip'])[0]
        self.dwell_grads = tf.clip_by_global_norm(tf.gradients(self.dwell_loss, dwell_vs), pd['grad_clip'])[0]
        with tf.variable_scope('opt_click'):
            click_optimizer = tf.train.AdamOptimizer(pd['lr_q'])
            self.click_opt = click_optimizer.apply_gradients(zip(self.click_grads, click_vs),
                                                             global_step=self.global_step)
        with tf.variable_scope('opt_dwell'):
            dwell_optimizer = tf.train.AdamOptimizer(pd['lr_q'])
            self.dwell_opt = dwell_optimizer.apply_gradients(zip(self.dwell_grads, dwell_vs),
                                                             global_step=self.global_step)

        # compute ||w_click*grad(L_click) + w_click*grad(L_dwell)|| as reward of DDPG-network
        self.m_click_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="main_click/q")
        self.m_click_params.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_click/feat_embedding'))
        self.t_click_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_click/q")
        self.t_click_params.extend(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_click/feat_embedding'))
        self.m_dwell_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="main_dwell/q")
        self.m_dwell_params.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_dwell/feat_embedding'))
        self.t_dwell_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_dwell/q")
        self.t_dwell_params.extend(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_dwell/feat_embedding'))

        # loss of every example
        self.weighted_click_loss = tf.reshape(tf.square(click_diff), [-1], name="weighted_click_loss")
        self.weighted_dwell_loss = tf.reshape(tf.square(dwell_diff), [-1], name="weighted_dwell_loss")

        # ||w_click*grad(L_click) + w_click*grad(L_dwell)||
        start = time.time()
        grad_batch = 1
        # grad_batch = 5
        self.grad_norm = [0.0] * grad_batch
        self.click_norm = [0.0] * grad_batch
        self.dwell_norm = [0.0] * grad_batch
        for i in range(grad_batch):
            idx = self.ph_loss_pos[i]
            # w1*L1,clip_norm=1
            click_grads, click_norm = tf.clip_by_global_norm(
                tf.gradients(ys=self.weighted_click_loss[idx], xs=click_vs), 1.0)
            # w2*L2,clip_norm=1
            dwell_grads, dwell_norm = tf.clip_by_global_norm(
                tf.gradients(ys=self.weighted_dwell_loss[idx], xs=dwell_vs), 1.0)
            # self.grad_norm[i] = tf.global_norm(click_grads + dwell_grads)
            weighted_click_grads = [self.ph_click_weight[idx] * x for x in click_grads]
            weighted_dwell_grads = [self.ph_dwell_weight[idx] * x for x in dwell_grads]
            self.grad_norm[i] = tf.global_norm(weighted_click_grads + weighted_dwell_grads)
            self.click_norm[i] = click_norm
            self.dwell_norm[i] = dwell_norm
        print
        ",weighted_grads_", i, ",time:", time.time() - start, ",res:", resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024

        alpha = pd['double_networks_sync_step']
        self.sync_click_op = [tf.assign(t, (1.0 - alpha) * t + alpha * m) for t, m in
                              zip(self.t_click_params, self.m_click_params)]
        self.sync_dwell_op = [tf.assign(t, (1.0 - alpha) * t + alpha * m) for t, m in
                              zip(self.t_dwell_params, self.m_dwell_params)]


    def create_graph(self, scope):
        global g_training
        # scope:main/target
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # 必须设置zero_pad=False，这是因为PartitionedVariable不支持相关操作
            feat_dict = get_embeddings(pd['uid_size'],
                                       pd['emb_dim'],
                                       scope='feat_embedding',
                                       zero_pad=False,
                                       partitioner=tf.fixed_size_partitioner(pd['embed_dict_part'], 0))
            # obj-related params
            with tf.variable_scope("q"):
                n_batch = pd['batch_size']
                embed_dim = pd['emb_dim']

                #
                user_embed = tf.nn.embedding_lookup(feat_dict, self.sph_user, sp_weights=None,
                                                           partition_strategy='div', combiner='mean')
                user_field = tf.reshape(user_embed, shape=[n_batch, embed_dim])

                step_a = tf.reshape(user_field, shape=[n_batch, -1])

                step_w = tf.identity(tf.reshape(weight_field, shape=[n_batch, -1]), name="weight_embed")

                step_s = tf.identity(tf.concat([step_a, step_w], axis=1), name='gru_x')




                q = tf.layers.dense(step_s, 1)
                print
                'q.shape:', q.shape
                print
                'doc_embed.shape:', doc_embed.shape
        return doc_embed, q

    # call for temporal-diffenence learning
    def target_click_q(self, sess, ph_dict):
        return sess.run(self.click_tq, feed_dict={
            self.sph_user: ph_dict['user']})

    def target_dwell_q(self, sess, ph_dict):
        return sess.run(self.dwell_tq, feed_dict={
            self.sph_user: ph_dict['user']})

    # call for evaluating value networks
    def main_click_q(self, sess, ph_dict):
        return sess.run(self.click_mq, feed_dict={
            self.sph_user: ph_dict['user']})

    def main_dwell_q(self, sess, ph_dict):
        return sess.run(self.dwell_mq, feed_dict={
            self.sph_user: ph_dict['user']})


    # call for learning from data
    def learn_click(self, sess, ph_dict):
        loss, _, _ = sess.run([self.click_loss, self.click_grads, self.click_opt],
                              feed_dict={self.ph_click_nq: ph_dict["next_click_q"],
                                         self.ph_click_reward: ph_dict["click_reward"],
                                         self.sph_user: ph_dict['user']})
        global g_click_loss_sum, g_click_loss_cnt
        g_click_loss_sum += loss
        g_click_loss_cnt += 1

    def learn_dwell(self, sess, ph_dict):
        loss, _, _ = sess.run([self.dwell_loss, self.dwell_grads, self.dwell_opt],
                              feed_dict={self.ph_dwell_nq: ph_dict["next_dwell_q"],
                                         self.ph_dwell_reward: ph_dict["dwell_reward"],
                                         self.sph_user: ph_dict['user']})
        global g_dwell_loss_sum, g_dwell_loss_cnt
        g_dwell_loss_sum += loss
        g_dwell_loss_cnt += 1

    def compute_grad_norm(self, sess, ph_dict):
        grad_norm, click_norm, dwell_norm = sess.run([self.grad_norm, self.click_norm, self.dwell_norm],
                                                     feed_dict={self.ph_click_nq: ph_dict["next_click_q"],
                                                                self.ph_loss_pos: ph_dict["loss_pos"],
                                                                self.ph_dwell_nq: ph_dict["next_dwell_q"],
                                                                self.sph_clk_seq: ph_dict['clk_seq'],
                                                                self.ph_click_reward: ph_dict["click_reward"],
                                                                self.ph_dwell_reward: ph_dict["dwell_reward"],
                                                                self.ph_click_weight: ph_dict["click_weight"],
                                                                self.ph_dwell_weight: ph_dict["dwell_weight"],
                                                                self.sph_user: ph_dict['user']})
        return grad_norm, click_norm, dwell_norm




def sigmoid(x):
    return 1.0 / (1.0 + math.exp(max(min(-x, 1e4), -1e4)))

#3
def handle(sess, pnet, qnet, sess_data, obj_model):

    global g_rb
    g_rb.save(sess_data)
    while g_rb.has_batch():
        user, weight, click_rwd, dwell_rwd = g_rb.next_batch()
        phd = {}
        phd['user'] = np.array(user).reshape(pd['batch_size'])
        phd['weight'] = weight
        click_weight = []
        dwell_weight = []
        [click_weight.append(i[0]) for i in weight]
        [dwell_weight.append(i[1]) for i in weight]
        phd['click_weight'] = click_weight
        phd['dwell_weight'] = dwell_weight
        phd['click_reward'] = click_rwd
        phd['dwell_reward'] = dwell_rwd
        global g_batch_counter, g_training
        print
        datetime.datetime.now(), 'start to handle batch', g_batch_counter
        if g_training:
            g_batch_counter += 1

            ###train objnetwork
            start = time.time()
            next_click_q = obj_model.target_click_q(sess, phd)#预测下一个q值
            next_click_q = np.append(next_click_q[:, 1:],
                                     np.array([[0] for i in range(pd['batch_size'])], dtype=np.float32), 1)
            phd["next_click_q"] = next_click_q
            obj_model.learn_click(sess, phd)
            print
            "click_model train cost:", time.time() - start

            start = time.time()
            next_dwell_q = obj_model.target_dwell_q(sess, phd)
            next_dwell_q = np.append(next_dwell_q[:, 1:],
                                     np.array([[0] for i in range(pd['batch_size'])], dtype=np.float32), 1)
            phd["next_dwell_q"] = next_dwell_q
            obj_model.learn_dwell(sess, phd)
            print
            "dwell_model train cost:", time.time() - start

            # 0.1 ratio train weight network
            rand_r = random.randint(1, 100)
            if rand_r == 1:
                start = time.time()
                grad_reward = []
                click_norm = []
                dwell_norm = []
                for p in range(0, pd["batch_size"]):
                    step_size = 1
                    # step_size = 5
                    phd["loss_pos"] = range(p * step_size, (p + 1) * step_size)
                    sub_grad, sub_click, sub_dwell = obj_model.compute_grad_norm(sess, phd)#输出梯度
                    grad_reward.extend(sub_grad)
                    click_norm.extend(sub_click)
                    dwell_norm.extend(sub_dwell)
                # qnet target:sum_grad_reward maximize!
                grad_reward = [-1.0 * x for x in grad_reward]
                phd["grad_reward"] = grad_reward
                print
                "grad_reward cost:", time.time() - start
                # print "click_norm:", click_norm
                # print "dwell_norm:", dwell_norm
                # print "grad_reward:", grad_reward

                # train Value Network
                start = time.time()
                next_q = qnet.target_q(sess, phd)
                next_q = np.append(next_q[:, 1:], np.array([[0] for i in range(pd['batch_size'])], dtype=np.float32), 1)
                phd['next_q'] = next_q
                qnet.learn(sess, phd)
                print
                "qnet train cost:", time.time() - start

                # train Policy Network
                if g_batch_counter % 1 == 0:
                    start = time.time()
                    action = pnet.act_for_train(sess, phd)
                    phd['action'] = action
                    pg = qnet.pg(sess, phd)
                    phd['a_grads'] = pg
                    pnet.learn(sess, phd)
                    print
                    "pnet train cost:", time.time() - start
            if g_batch_counter % (pd['double_networks_sync_freq'] * 10) == 0:
                print
                'Run soft replacement for weight networks...'
                sess.run(pnet.sync_op)
                sess.run(qnet.sync_op)
            if g_batch_counter % pd['double_networks_sync_freq'] == 0:
                print
                'Run soft replacement for click networks and dwell networks...'
                sess.run(obj_model.sync_click_op)
                sess.run(obj_model.sync_dwell_op)

        else:
            g_batch_counter += 1
            # Evaluate Click Network
            click_out = obj_model.main_click_q(sess, phd).reshape([-1])
            global g_working_mode
            print('%s\t%s\t%s' % (
                click_out[i], click_rwd[i], dwell_rwd[i],))
    else:
        pass

    # Evaluate Dwell Network
    dwell_out = obj_model.main_dwell_q(sess, phd).reshape([-1])
    for i in range(len(dwell_rwd)):
        if 'local_predict' == g_working_mode:
            print('%s\t%s\t%s' % (
                dwell_out[i], click_rwd[i], dwell_rwd[i],
                ))
        elif 'local_run_for_deepx_converter' == g_working_mode:
            pass
        else:
            pass

    # Evaluate Critic Network
    q_out = qnet.main_q(sess, phd).reshape([-1])

    # w1*q1 + w2*q2

    # Evaluate Policy Network
    action = pnet.act_for_eval(sess, phd)
    action = action.reshape([pd["batch_size"], pd["num_objs"]])
    for i in range(len(click_rwd)):
        if 'local_predict' == g_working_mode:
            print('%s\t%s\t%s' % (sigmoid(click_out[i] * 1.070340 + dwell_out[i] * 1.745755),
                                    click_rwd[i], dwell_rwd[i]))
        elif 'local_run_for_deepx_converter' == g_working_mode:
            pass
        else:
            1



def do_train(sess, pnet, qnet, obj_model):
    print
    'start do train...'
    ######################################################
    #                 这里需要写上训练文件的地址              #
    #                                                    #
    ######################################################
    filelist = train_file
    print
    'filelist: ', filelist
    train_sess_cnt = 0
    train_file_idx = 0
    global g_training
    for fname in filelist:
        train_file_idx += 1
        print
        datetime.datetime.now(), fname
        sys.stdout.flush()
        # train_ratio = 5
        if (g_training and train_file_idx % 10 >= 5) or (not g_training and train_file_idx % 100 <= 95):
            # if (g_training and train_file_idx % 5000 != 1) or (not g_training and train_file_idx % 5000 != 1):
            print
            'to speed up, skip...'
            continue
        part = fname.split('/')[-1][:-3]
        if not os.path.exists('train_data.txt'):
            continue

        for data in read_data('train_data.txt', pd['clk_seq_len'], pd['user_field_num'], pd['doc_field_num'],
                              pd['con_field_num'], pd['feat_prime'], pd["history_doc_field_num"]):
            # TODO
            handle(sess, pnet, qnet, data, obj_model)
            train_sess_cnt += 1
            global g_p_loss_sum, g_p_loss_cnt, g_v_loss_sum, g_v_loss_cnt
            global g_click_loss_sum, g_click_loss_cnt, g_dwell_loss_sum, g_dwell_loss_cnt
            if g_training and train_sess_cnt % 1000 == 0:
                print
                '------PolicyNetwork Global Step: ', sess.run(pnet.global_step)
                print
                '------ValueNetwork Global Step: ', sess.run(qnet.global_step)
                print
                '------Train Session Count: ', train_sess_cnt
                print
                '------Average PolicyNetwork Loss: ', g_p_loss_sum / (g_p_loss_cnt + 1e-6)
                print
                '------Average ValueNetwork Loss: ', g_v_loss_sum / (g_v_loss_cnt + 1e-6)
                print
                '------Average ClickNetwork Loss: ', g_click_loss_sum / (g_click_loss_cnt + 1e-6)
                print
                '------Average DwellNetwork Loss: ', g_dwell_loss_sum / (g_dwell_loss_cnt + 1e-6)

                g_p_loss_sum = 0.
                g_p_loss_cnt = 0
                g_v_loss_sum = 0.
                g_v_loss_cnt = 0
                g_click_loss_sum = 0.
                g_click_loss_cnt = 0
                g_dwell_loss_sum = 0.
                g_dwell_loss_cnt = 0

        if not g_training:
            print
            'close new file pred.txt'

        # print 'fast test, exit!!!!'
        # break


#2
def local_run():
    global_step = tf.train.get_or_create_global_step()#创建一个全局的步数
    sess = tf.Session()#为了控制,和输出文件的执行的语句
    # build networks
    pnet = PolicyNetwork(global_step)#实例化actor
    print
    '--------'
    qnet = ValueNetwork(global_step)#实例化critic
    obj_model = ObjNetwork(global_step)#实例化network

    saver = tf.train.Saver(max_to_keep=1)#保存和恢复都需要实例化一个 tf.train.Saver
    g_init_op = tf.global_variables_initializer()#初始化模型的参数
    if os.path.exists('./ckpt'):
        model_file = tf.train.latest_checkpoint('ckpt/')#查找最新保存的checkpoint文件的文件名
        saver.restore(sess, model_file)#重载模型的参数，继续训练或用于测试数据
    else:
        sess.run(g_init_op)#可以获得你要得知的运算结果, 或者是你所要运算的部分
        os.system('mkdir ckpt')
    print
    '>>>local model...'
    global g_p_loss_sum, g_p_loss_cnt, g_v_loss_sum, g_v_loss_cnt
    global g_click_loss_sum, g_click_loss_cnt, g_click_loss_sum, g_click_loss_cnt
    pull_cnt = 0
    for data in read_data('sample.in', pd['clk_seq_len'], pd['user_field_num'], pd['doc_field_num'],
                          pd['con_field_num'], pd['feat_prime'], pd["history_doc_field_num"]):
        handle(sess, pnet, qnet, data, obj_model)
        pull_cnt += 1
        if g_training and pull_cnt % 500 == 0:
            print
            '>>>Average PolicyNetwork Loss: %f' % (g_p_loss_sum / (g_p_loss_cnt + 1e-6))
            print
            '>>>Average ValueNetwork Loss: %f' % (g_v_loss_sum / (g_v_loss_cnt + 1e-6))
            print
            '>>>Average ClickNetwork Loss: %f' % (g_click_loss_sum / (g_click_loss_cnt + 1e-6))
            print
            '>>>Average DwellNetwork Loss: %f' % (g_dwell_loss_sum / (g_dwell_loss_cnt + 1e-6))
    saver.save(sess, 'ckpt/auto_ddpg.ckpt')#训练循环中，定期调用这个方法，向文件夹中写入包含当前模型中所有可训练变量的 checkpoint 文件




#1
if __name__ == '__main__':
    # g_working_mode = 'local_train'
    # g_rb = RB(pd['batch_size'], pd['user_field_num'], pd['doc_field_num'],
    #           pd['con_field_num'], pd["history_doc_field_num"])
    # commander = {
    #     'local_train': local_run,
    #     'local_predict': local_run,
    # }
    # g_training = (g_working_mode == 'local_train' or g_working_mode == 'distributed_train')
    # print
    # '>>> working_model:', g_working_mode
    # print
    # '>>> is_training:', g_training
    # commander[g_working_mode]()
    local_run()