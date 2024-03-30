import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from gae.infrastructure import pytorch_util as ptu
from gae.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=True,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # Action NN
        self.action_mlp = ptu.build_critic_mlp(input_size=self.ob_dim,
                                        output_size=4,
                                        n_layers=self.n_layers,
                                        size=self.size)
        self.action_mlp.to(ptu.device)
        self.action_optimizer = optim.Adam(self.action_mlp.parameters(),
                                    self.learning_rate)

        # Betting NN
        self.bet_mlp = ptu.build_critic_mlp(input_size=self.ob_dim,
                                    output_size=1,
                                    n_layers=self.n_layers, size=self.size)
        self.logstd = nn.Parameter(
            torch.zeros(1, dtype=torch.float32, device=ptu.device)
        )
        self.bet_mlp.to(ptu.device)
        self.logstd.to(ptu.device)
        self.bet_optimizer = optim.Adam(
            itertools.chain([self.logstd], self.bet_mlp.parameters()),
            self.learning_rate
        )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: get this from hw1 or hw2
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        action_distribution = self.forward(observation, discrete=True)
        action = action_distribution.sample()

        action_taken = torch.zeros([4])
        action_taken[action] = 1
        observation[0,0:4] = action_taken.to(int)
        bet_distribution = self.forward(observation, discrete=False)
        bet = bet_distribution.sample()
        return np.concatenate([ptu.to_numpy(action), ptu.to_numpy(bet.squeeze(1))], axis=0)
        # observation = ptu.from_numpy(observation)
        # action = self.forward(observation)
        # action = ptu.to_numpy(action)
        # act_choice = np.argmax(action[:-1])
        # raise_amt = action[-1]
        # return np.array([act_choice, raise_amt])

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        return None

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor, discrete=False):
        # return self.mean_net(observation)
        # logits = self.action_mlp(observation)
        # action_distribution = distributions.Categorical(logits=logits)
        # return action_distribution
        if discrete:
            logits = self.action_mlp(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.bet_mlp(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution

#####################################################
#####################################################


class MLPPolicyAC(MLPPolicy):
    def update(self, observations, actions, adv_n=None):
        # TODO: update the policy and return the loss
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)

        ob_tensor = torch.Tensor(observations)
        ac_distribution = self.forward(ob_tensor)
        log_pi = ac_distribution.log_prob(actions)
        loss = -torch.mean(log_pi*(adv_n-0.01))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()