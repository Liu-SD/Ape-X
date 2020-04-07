import _pickle as pickle
import os
from multiprocessing import Process, Queue
import queue

import zmq
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

import utils
from memory import BatchStorage
from wrapper import make_atari, wrap_atari_dqn
from model import DuelingDQN
from arguments import argparser
from datetime import datetime


def get_environ():
    actor_id = int(os.environ.get('ACTOR_ID', '-1'))
    n_actors = int(os.environ.get('N_ACTORS', '-1'))
    replay_ip = os.environ.get('REPLAY_IP', '-1')
    learner_ip = os.environ.get('LEARNER_IP', '-1')
    assert (actor_id != -1 and n_actors != -1)
    assert (replay_ip != '-1' and learner_ip != '-1')
    return actor_id, n_actors, replay_ip, learner_ip


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

def vector_exploration(args, actor_id, n_actors, replay_ip, param_queue):
    ctx = zmq.Context()
    batch_socket = ctx.socket(zmq.DEALER)
    batch_socket.setsockopt(zmq.IDENTITY, pickle.dumps('actor-{}'.format(actor_id)))
    batch_socket.connect('tcp://{}:51001'.format(replay_ip))
    outstanding = 0

    writer = SummaryWriter(comment="-{}-actor{}".format(args.env, actor_id))

    num_envs = args.num_envs_per_worker
    envs = [wrap_atari_dqn(make_atari(args.env), args) for _ in range(num_envs)]

    if args.seed is not None:
        seeds = args.seed + actor_id * num_envs + np.arange(num_envs)
        utils.set_global_seeds(seeds[0], use_torch=True)
        for seed, env in zip(seeds, envs):
            env.seed(int(seed))

    model = DuelingDQN(envs[0], args)
    model = torch.jit.trace(model, torch.zeros((1,4,84,84)))
    _actor_id = np.arange(num_envs) + actor_id * num_envs
    n_actors = n_actors * num_envs
    epsilons = args.eps_base ** (1 + _actor_id / (n_actors - 1) * args.eps_alpha)
    storages = [BatchStorage(args.n_steps, args.gamma) for _ in range(num_envs)]

    param = param_queue.get(block=True)
    model.load_state_dict(param)
    param = None
    print("%d: Received First Parameter!"%actor_id)

    actor_idx = 0
    tb_idx = 0
    episode_rewards = np.array([0] * num_envs)
    episode_lengths = np.array([0] * num_envs)
    states = np.array([env.reset() for env in envs])
    tb_dict = {key: [] for key in ['episode_reward', 'episode_length']}
    step_t = time.time()
    ref_t = 0
    sim_t = 0
    while True:
        if actor_idx * num_envs * n_actors <= args.initial_exploration_samples: # initial random exploration
            random_idx = np.arange(num_envs)
        else:
            random_idx, = np.where(np.random.random(num_envs) <= epsilons)
        _t = time.time()
        with torch.no_grad():
            states_tensor = torch.tensor(states, dtype=torch.float32)
            q_values = model(states_tensor).detach().numpy()
        ref_t += time.time() - _t
        actions = np.argmax(q_values, 1)
        actions[random_idx] = np.random.choice(envs[0].action_space.n, len(random_idx))

        for i, (state, q_value, action, env, storage) in enumerate(zip(states, q_values, actions, envs, storages)):
            _t = time.time()
            next_state, reward, done, _ = env.step(action)
            sim_t += time.time() - _t
            storage.add(np.array(state), reward, action, done, q_value, _t, episode_lengths[i])
            states[i] = next_state
            episode_rewards[i] += reward
            episode_lengths[i] += 1
            if done or episode_lengths[i] == args.max_episode_length:
                states[i] = env.reset()
                tb_idx += 1
                tb_dict["episode_reward"].append(episode_rewards[i])
                tb_dict["episode_length"].append(episode_lengths[i])
                episode_rewards[i] = 0
                episode_lengths[i] = 0
                if tb_idx % args.tb_interval == 0:
                    writer.add_scalar('actor/episode_reward_mean', np.mean(tb_dict['episode_reward']), tb_idx)
                    writer.add_scalar('actor/episode_reward_max', np.max(tb_dict['episode_reward']), tb_idx)
                    writer.add_scalar('actor/episode_reward_min', np.min(tb_dict['episode_reward']), tb_idx)
                    writer.add_scalar('actor/episode_reward_std', np.std(tb_dict['episode_reward']), tb_idx)
                    writer.add_scalar('actor/episode_length_mean', np.mean(tb_dict['episode_length']), tb_idx)
                    tb_dict['episode_reward'].clear()
                    tb_dict['episode_length'].clear()
                    writer.add_scalar('actor/step_time', (time.time() - step_t) / np.sum(episode_lengths), tb_idx)
                    writer.add_scalar('actor/step_inference_time', ref_t / np.sum(episode_lengths), tb_idx)
                    writer.add_scalar('actor/step_simulation_time', sim_t / np.sum(episode_lengths), tb_idx)
                    ref_t = 0
                    sim_t = 0
                    step_t = time.time()

        actor_idx += 1

        if actor_idx % args.update_interval == 0:
            try:
                param = param_queue.get(block=False)
                model.load_state_dict(param)
                print("%d: Updated Parameter.."%actor_id)
            except queue.Empty:
                pass

        if sum(len(storage) for storage in storages) >= args.send_interval * num_envs:
            batch = []
            prios = []
            for storage in storages:
                _batch, _prios = storage.make_batch()
                batch.append(_batch)
                prios.append(_prios)
                storage.reset()
            batch = [np.concatenate(v) for v in zip(*batch)]
            prios = np.concatenate(prios)
            data = pickle.dumps((batch, prios))
            batch, prios = None, None
            while outstanding >= args.max_outstanding:
                batch_socket.recv()
                outstanding -= 1
            batch_socket.send(data, copy=False)
            outstanding += 1
            # print(f"{actor_id}: send  batch {len(data)/1024} kb")

def main():
    actor_id, n_actors, replay_ip, learner_ip = get_environ()
    args = argparser()
    param_queue = Queue(maxsize=3)

    procs = [
        Process(target=vector_exploration, args=(args, actor_id, n_actors, replay_ip, param_queue)),
        Process(target=recv_param, args=(learner_ip, actor_id, param_queue)),
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return True


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
