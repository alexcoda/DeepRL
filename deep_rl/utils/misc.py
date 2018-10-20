#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import os
import datetime
import torch
import time
from .torch_utils import *
try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path
from ..agent import DQN_agent as DQN_agent
from ..component import PixelAtari

def run_steps(agent):
    random_seed()
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    if config.load_model is not None:
        config.logger.info("Loading model {}".format(config.load_model))
        agent.load(config.load_model)
    env_changed = False
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag))
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            agent.episode_rewards = []
            config.logger.info('total steps %d, returns %.2f/%.2f/%.2f/%.2f (mean/median/min/max), %.2f steps/s' % (
                agent.total_steps, np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards),
                config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if agent.total_steps>0 and agent.total_steps%config.max_steps==0:
            if config.env_difficulty == 0 and not env_changed:
                config.env_difficulty = 1
                config.logger.info("Environment Difficulty Changed from %d to %d"%(0,config.env_difficulty))
                env_changed = True
            elif config.env_difficulty == 1:
                config.env_difficulty = 0
                config.logger.info("Environment Difficulty Changed from %d to %d"%(1,config.env_difficulty))
            else:
                agent.close()
                break
            config.task_fn = lambda: PixelAtari(
                config.name, frame_skip=4, history_length=config.history_length,
                use_new_atari_env=config.use_new_atari_env, env_mode=config.env_mode,
                env_difficulty=config.env_difficulty)
            config.eval_env = PixelAtari(
                config.name, frame_skip=4, history_length=config.history_length,
                episode_life=False, use_new_atari_env=config.use_new_atari_env,
                env_mode=config.env_mode, env_difficulty=config.env_difficulty)
            current_state_dict = agent.network.state_dict()
            agent.actor = DQN_agent.DQNActor(config)
            agent.actor.set_network(agent.network)
            # agent.close()
            # break
        agent.step()

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()

class Batcher:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]
