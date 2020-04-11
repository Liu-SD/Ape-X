"""
Module for evaluator in Ape-X.
"""
import _pickle as pickle
import os
from multiprocessing import Process, Queue

import zmq
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import utils
from wrapper import make_atari, wrap_atari_dqn
from model import DuelingDQN
from arguments import argparser
from datetime import datetime


def get_environ():
    learner_ip = os.environ.get('LEARNER_IP', '-1')
    assert learner_ip != '-1'
    return learner_ip


def connect_param_socket(ctx, param_socket, learner_ip, actor_id):
    socket = ctx.socket(zmq.REQ)
    socket.connect("tcp://{}:52002".format(learner_ip))
    socket.send(pickle.dumps((actor_id, 1)))
    socket.recv()
    param_socket.connect('tcp://{}:52001'.format(learner_ip))
    socket.send(pickle.dumps((actor_id, 2)))
    socket.recv()
    print("Successfully connected to learner!")
    socket.close()


def recv_param(learner_ip, actor_id, param_queue):
    ctx = zmq.Context()
    param_socket = ctx.socket(zmq.SUB)
    param_socket.setsockopt(zmq.SUBSCRIBE, b'')
    param_socket.setsockopt(zmq.CONFLATE, 1)
    connect_param_socket(ctx, param_socket, learner_ip, actor_id)
    while True:
        data = param_socket.recv(copy=False)
        param = pickle.loads(data)
        param_queue.put(param)


def exploration(args, actor_id, param_queue):
    writer = SummaryWriter(comment="-{}-eval".format(args.env))

    args.clip_rewards = False
    args.episode_life = False
    env = make_atari(args.env)
    env = wrap_atari_dqn(env, args)

    seed = args.seed + actor_id
    utils.set_global_seeds(seed, use_torch=True)
    env.seed(seed)

    model = DuelingDQN(env, args)

    param = param_queue.get(block=True)
    model.load_state_dict(param)
    param = None
    print("Received First Parameter!")

    episode_reward, episode_length, episode_idx = 0, 0, 0
    state = env.reset()
    tb_dict = {k: [] for k in ['episode_reward', 'episode_length']}
    while True:
        action, _ = model.act(torch.FloatTensor(np.array(state)), 0.)
        next_state, reward, done, _ = env.step(action)

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done or episode_length == args.max_episode_length:
            state = env.reset()
            tb_dict["episode_reward"].append(episode_reward)
            tb_dict["episode_length"].append(episode_length)
            episode_reward = 0
            episode_length = 0
            episode_idx += 1
            param = param_queue.get()
            model.load_state_dict(param)
            print(f"{datetime.now()} Updated Parameter..")

            if (episode_idx * args.num_envs_per_worker) % args.tb_interval == 0:
                writer.add_scalar('evaluator/episode_reward_mean', np.mean(tb_dict['episode_reward']), episode_idx)
                writer.add_scalar('evaluator/episode_reward_max', np.max(tb_dict['episode_reward']), episode_idx)
                writer.add_scalar('evaluator/episode_reward_min', np.min(tb_dict['episode_reward']), episode_idx)
                writer.add_scalar('evaluator/episode_reward_std', np.std(tb_dict['episode_reward']), episode_idx)
                writer.add_scalar('evaluator/episode_length_mean', np.mean(tb_dict['episode_length']), episode_idx)
                tb_dict['episode_reward'].clear()
                tb_dict['episode_length'].clear()


def main():
    learner_ip = get_environ()
    args = argparser()
    param_queue = Queue(maxsize=3)

    procs = [
        Process(target=exploration, args=(args, -1, param_queue)),
        Process(target=recv_param, args=(learner_ip, -1, param_queue)),
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return True


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
