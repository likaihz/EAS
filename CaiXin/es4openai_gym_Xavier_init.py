"""
Simple code for Distributed ES proposed by OpenAI.
Based on this paper: Evolution Strategies as a Scalable Alternative to Reinforcement Learning
Details can be found in : https://arxiv.org/abs/1703.03864
Visit more on morvanzhou tutorial site: https://morvanzhou.github.io/tutorials/

Reconstruct based on morvanzhou's code: https://github.com/MorvanZhou/Evolutionary-Algorithm/blob/master/tutorial-contents/Using%20Neural%20Nets/Evolution%20Strategy%20with%20Neural%20Nets.py
Only for studying Distributed ES and reinforcement learning
"""
import numpy as np
import gym
import multiprocessing as mp
import time
import os
import pickle
import argparse
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser()
parser.add_argument('--to_render', type=str2bool, default=True, metavar='if test in render',
                    help="test in render mode (default: True)")
parser.add_argument('--retrain', type=str2bool, default=False, metavar='train again',
                    help="Train model again (default: False)")
parser.add_argument('--n_kid', type=int, default=32, metavar='n_kid', 
                    help='The number of the kids')
parser.add_argument('--n_generation', type=int, default=5000, metavar='n_generation', 
                    help='The number of the generation(EPOCH)')
parser.add_argument('--test_epoch', type=int, default=3, metavar='test_epoch', 
                    help='The number of the test epoch(EPOCH)')
parser.add_argument('--game', type=int, default=1, metavar='game', 
                    help='The No of game')
parser.add_argument('--save_path', type=str, default='model.pkl', metavar='save_path', 
                    help='The directory of the inout model')
args = parser.parse_args()

TEST_EPOCH = args.test_epoch
TO_RENDER = args.to_render
N_KID = args.n_kid          # half of the training population
N_GENERATION = args.n_generation         # training step
LR = .05                    # learning rate
SIGMA = .05                 # mutation strength or step size
N_CORE = mp.cpu_count()-1
SAVE_PATH = args.save_path
CONFIG = [
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=699),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Pendulum-v0",
         n_feature=3, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180),
    dict(game="MountainCarContinuous-v0",
         n_feature=2, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=30),
][args.game]    # choose your game

def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling

def get_reward(model, env, seed_and_id=None):
    # perturb parameters using seed
    params = model.net_params
    shapes = model.net_shapes
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        for i, shape in enumerate(shapes):
            w = np.random.randn(*shape)# * np.sqrt(1 / shape[0])
            b = np.random.randn(1, shape[1])# * np.sqrt(1 / shape[0])
            params[i * 2] += sign(k_id) * model.sigma * w
            params[i * 2 + 1] += sign(k_id) * model.sigma * b

    # run episode
    s = env.reset()
    ep_r = 0.
    for step in range(model.ep_max_step):
        a = model.choose_action(s)
        s, r, done, _ = env.step(a)
        # mountain car's reward can be tricky
        if env.spec._env_name == 'MountainCar' and s[0] > -0.1: r = 0.
        ep_r += r
        if done: break
    return ep_r

class SGD(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = [np.zeros_like(param).astype(np.float32) for param in params]
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients, w):
        lr_gradients = []
        for i, gradient in enumerate(gradients):
            self.v[i] = self.momentum * self.v[i] + (1. - self.momentum) * gradient
            lr_gradients.append(self.lr * self.v[i])
        return lr_gradients

class DistributedESRL:
    def __init__(self, n_kid, n_generation, config, n_core = N_CORE, learning_rate=LR, sigma=SIGMA):
        self.n_actions = config['n_action']
        self.n_features = config['n_feature']
        self.continuous_a = config['continuous_a']
        self.ep_max_step = config['ep_max_step']
        self.eval_threshold = config['eval_threshold']
        self.n_core = n_core
        self.sigma = sigma
        self.n_kid = n_kid                  # half of the training population
        self.n_generation = n_generation    # training step
        self.lr = learning_rate     # 学习率
        # self.gamma = reward_decay   # reward 递减率        
        self.net_shapes, self.net_params = self._build_net()   
        self.optimizer = SGD(self.net_params, self.lr)

    def _build_net(self):
        def linear(n_in, n_out):  # network linear layer
            w = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in) * 0.1
            b = np.zeros((1, n_out))
            return (w, b)
        layers_dims = [(self.n_features, 30), (30, 20), (20, self.n_actions)]
        params = []
        for layer_dims in layers_dims:
            params += linear(*layer_dims)
        return layers_dims, params

    def choose_action(self, x):
        x = x[np.newaxis, :]
        x = np.tanh(x.dot(self.net_params[0]) + self.net_params[1])
        x = np.tanh(x.dot(self.net_params[2]) + self.net_params[3])
        x = x.dot(self.net_params[4]) + self.net_params[5]
        if not self.continuous_a[0]: 
            return np.argmax(x, axis=1)[0]                 # for discrete action
        else: 
            return self.continuous_a[1] * np.tanh(x)[0]    # for continuous action

    def train(self, env):
        # utility instead reward for update parameters (rank transformation)
        base = self.n_kid * 2    # *2 for mirrored sampling
        rank = np.arange(1, base + 1)
        util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
        utility = util_ / util_.sum() - 1 / base

        # training
        pool = mp.Pool(processes=self.n_core)
        mar = None      # moving average reward
        for g in range(self.n_generation):
            t0 = time.time()
            kid_rewards = self.step(env, utility, pool)
            # test trained net without noise
            net_r = get_reward(self, env, None,)
            mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
            print(
                'Gen: ', g,
                '| Net_R: %.1f' % mar,
                '| Kid_avg_R: %.1f' % kid_rewards.mean(),
                '| Gen_T: %.2f' % (time.time() - t0),)
            if mar >= self.eval_threshold: break

    def step(self, env, utility, pool):
        # pass seed instead whole noise matrix to parallel will save your time
        noise_seed = np.random.randint(0, 2 ** 32 - 1, size=self.n_kid, dtype=np.uint32).repeat(2)    # mirrored sampling

        # distribute training in parallel
        jobs = [pool.apply_async(get_reward, (self, env, [noise_seed[k_id], k_id], )) for k_id in range(self.n_kid*2)]
        rewards = np.array([j.get() for j in jobs])
        kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward

        cumulative_update = [np.zeros_like(params) for params in self.net_params]       # initialize update values
        for ui, k_id in enumerate(kids_rank):
            np.random.seed(noise_seed[k_id])
            for i, shape in enumerate(self.net_shapes):
                w = np.random.randn(*shape)# * np.sqrt(1 / shape[0])
                b = np.random.randn(1, shape[1])# * np.sqrt(1 / shape[0])
                # reconstruct noise using seed
                cumulative_update[i * 2] += utility[ui] * sign(k_id) * w
                cumulative_update[i * 2 + 1] += utility[ui] * sign(k_id) * b

        gradients = self.optimizer.get_gradients(cumulative_update, (2*self.n_kid*self.sigma))
        for i, gradient in enumerate(gradients):
            self.net_params[i] += gradient
        return rewards

    def save_model(self, path):
        with open(path, 'wb') as model:
            pickle.dump((self.net_shapes, self.net_params), model)

    def load_model(self, path):
        with open(path, 'rb') as model:
            self.net_shapes, self.net_params = pickle.load(model)

def doNothing():
    pass

def buildAndTrainModel(env):
    retrain = args.retrain
    if retrain or not os.path.exists(SAVE_PATH):
        # train
        print("Train from scratch")
        model = DistributedESRL(N_KID, N_GENERATION, CONFIG)
        model.train(env)
        model.save_model(SAVE_PATH)
    else:
        model = DistributedESRL(N_KID, N_GENERATION, CONFIG)
        model.load_model(SAVE_PATH)
    return model

def printEnvInfo(env):
    print("actions: ", env.action_space) # 查看这个环境中可用的 action 有多少个
    print("observations: ",  env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
    print(env.observation_space.high)   # 查看 observation 最高取值
    print(env.observation_space.low)    # 查看 observation 最低取值

def testModel(model, env):
    # test
    print("\nTESTING....")
    count = 0
    steps = []
    rs = []
    if TO_RENDER:
        render = env.render
    else:
        render = doNothing
    while True:
        s = env.reset()
        tmp_rs = []
        for i in range(CONFIG['ep_max_step']):
            render()
            a = model.choose_action(s)
            s, r, done, _ = env.step(a)
            tmp_rs.append(r)
            if done:
                break
        steps.append(i)
        rs.append(sum(tmp_rs))
        count += 1
        if count == TEST_EPOCH:
            break
    env.close()
    print("AVG steps: {}, AVG reward: {}".format(np.average(steps), np.average(rs)))

if __name__ == "__main__":
    # init
    env = gym.make(CONFIG['game']).unwrapped
    printEnvInfo(env)
    model = buildAndTrainModel(env)
    testModel(model, env)