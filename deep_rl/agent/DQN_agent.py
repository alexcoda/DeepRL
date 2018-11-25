#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
import copy 

class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock:
            q_values, _ = self._network(config.state_normalizer(np.stack([self._state])))
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step(action)
        entry = [self._state, action, reward, next_state, int(done), info]
        self._total_steps += 1
        if done:
            next_state = self._task.reset()
        self._state = next_state
        return entry

class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor1 = DQNActor(config)
        # config2 = copy.deepcopy(config)
        config_dict = config.__dict__
        config2 = Config()
        config2.merge(config_dict=config_dict)
        config2.task_fn = config.task_fn2
        print(config2.__dict__)
        print("Initializing actor 2...")
        self.actor2 = DQNActor(config2)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor1.set_network(self.network)
        self.actor2.set_network(self.network)

        print(self.network)

        self.episode_reward = 0
        self.episode_reward2 = 0
        self.episode_rewards1 = []
        self.episode_rewards2 = []

        self.total_steps = 0
        self.total_steps2 = 0
        self.batch_indices = range_tensor(self.replay.batch_size)
        self.criterion_domain = torch.nn.MSELoss()

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor1)
        close_obj(self.actor2)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        q, _ = self.network(state)
        action = np.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        transitions = np.array(self.actor1.step())
        transitions2 = np.array(self.actor2.step())
        # print("actor 2 transitions", transitions2.shape)
        transitions = np.vstack([transitions,transitions2])
        experiences = []
        # print(transitions.shape)
        for i,(state, action, reward, next_state, done, _) in enumerate(transitions):
            if i==0:
                self.episode_reward += reward
                self.total_steps += 1
            else:
                self.episode_reward2 += reward
                self.total_steps2 += 1
            reward = config.reward_normalizer(reward)
            if done:
                if i==0:
                    self.episode_rewards1.append(self.episode_reward)
                    self.episode_reward = 0
                else:
                    self.episode_rewards2.append(self.episode_reward2)
                    self.episode_reward2 = 0
            experiences.append([state, action, reward, next_state, done, i])
        self.replay.feed_batch(experiences)

        if self.total_steps + self.total_steps2 > self.config.exploration_steps:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals, target_domains = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            q_next, _ = self.target_network(next_states)
            q_next = q_next.detach()
            if self.config.double_q:
                op_qvals, op_domain = self.network(next_states)
                best_actions = torch.argmax(op_qvals, dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()
            q, op_domains = self.network(states)
            q = q[self.batch_indices, actions]
            q_loss = (q_next - q).pow(2).mul(0.5).mean() 
            domain_loss = self.criterion_domain(op_domains, tensor(
                np.expand_dims(target_domains, axis=1)))
            loss = q_loss + domain_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
