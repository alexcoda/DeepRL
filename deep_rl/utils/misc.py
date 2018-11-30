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

def run_steps(agent):
    random_seed()
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    if config.load_model is not None:
        config.logger.info("Loading model {}".format(config.load_model))
        agent.load(config.load_model)
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/model-%s-%s-%s-%d.bin' % (agent_name, config.task_name, config.tag, agent.total_steps))
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards1):
            rewards1 = agent.episode_rewards1
            agent.episode_rewards1 = []
            config.logger.info('total steps %d, returns (0) %.2f/%.2f/%.2f/%.2f (mean/median/min/max), %.2f steps/s' % (
                agent.total_steps, np.mean(rewards1), np.median(rewards1), np.min(rewards1), np.max(rewards1),
                config.log_interval / (time.time() - t0)))
            rewards2 = agent.episode_rewards2
            agent.episode_rewards2 = []
            config.logger.info('total steps %d, returns (1) %.2f/%.2f/%.2f/%.2f (mean/median/min/max), %.2f steps/s' % (
                agent.total_steps2, np.mean(rewards2), np.median(rewards2), np.min(rewards2), np.max(rewards2),
                config.log_interval / (time.time() - t0)))
            config.logger.scalar_summary("MeanReward", np.mean(rewards), step=agent.total_steps)
            config.logger.scalar_summary("MedianReward", np.median(rewards), step=agent.total_steps)
            config.logger.scalar_summary("MinReward", np.min(rewards), step=agent.total_steps)
            config.logger.scalar_summary("MaxReward", np.max(rewards), step=agent.total_steps)
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()

def get_time_str():
    return datetime.datetime.now().strftime("%m%d-%H%M")

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
