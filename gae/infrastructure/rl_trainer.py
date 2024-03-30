from collections import OrderedDict
import pickle
import os
import sys
import time

import gymnasium as gym
from gymnasium import wrappers
import numpy as np
import torch
from gae.infrastructure import pytorch_util as ptu

from gae.infrastructure import utils

from gae.infrastructure.monitor import Monitor

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        utils.register_custom_env()
        self.env = gym.make(self.params['env_name'], num_rounds=1000)
        if 'env_wrappers' in self.params:
            # These operations are currently only for Atari envs
            self.env = Monitor(
                self.env,
                os.path.join(self.params['logdir'], "gym"),
                force=True,
                video_callable=(None if self.params['video_log_freq'] > 0 else False),
            )
            self.env = params['env_wrappers'](self.env)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')

        # self.env.seed(seed)

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes
        
        # Two player has obs as a tuple
        ob_dim = 19
        ac_dim = 1
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        self.fps = -1

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 1

        for itr in range(n_iter + 1):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # collect trajectories, to be used for training
            use_batchsize = self.params['batch_size']
            if itr==0:
                use_batchsize = self.params['batch_size_initial']
            paths, envsteps_this_batch = (
                self.collect_training_trajectories(
                    itr, initial_expertdata, collect_policy, use_batchsize)
            )

            self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            self.train_agent()

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, num_transitions_to_sample, save_expert_data_to_disk=False):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # TODO: get this from hw1 or hw2
        if itr == 0:
            num_transitions_to_sample = self.params['batch_size_initial']
        else:
            num_transitions_to_sample = self.params['batch_size']

#        print('Collecting train data...')
        paths, envsteps_this_batch = utils.sample_trajectories(
            self.env,
            collect_policy,
            num_transitions_to_sample,
            self.params['ep_len']
        )

        return paths, envsteps_this_batch

    def train_agent(self):
        # TODO: get this from hw1 or hw2
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            obs_batch, act_batch, rew_batch, nobs_batch, term_batch = self.agent.sample(self.params['train_batch_size'])
            train_log = self.agent.train(obs_batch, act_batch, rew_batch, nobs_batch, term_batch)
            all_logs.append(train_log)
        return all_logs