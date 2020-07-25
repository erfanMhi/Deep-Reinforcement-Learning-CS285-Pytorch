
import torch

import numpy as np
import torch.nn as nn

from torch import optim
from .base_policy import BasePolicy
from cs285.infrastructure.torch_utils import MLP, multivariate_normal_diag, convert_args_to_tensor
from torch.distributions import Categorical

class MLPPolicy(BasePolicy):

    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        learning_rate=1e-4,
        training=True,
        policy_scope='policy_vars',
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate 
        self.training = training

        # Build Torch Graph
        self.build_graph()


    ##################################

    def build_graph(self):
        # self.define_placeholders()
        self.define_forward_pass_parameters()
        if self.training:
            self.define_train_op()

    ##################################

    def define_forward_pass_parameters(self):
        # TODO implement this build_mlp function in tf_utils
        mean = MLP(self.ob_dim, output_size=self.ac_dim, n_layers=self.n_layers, size=self.size)
        logstd = torch.zeros(self.ac_dim, requires_grad=True)
        self.parameters = (mean, logstd)

    def _build_action_sampling(self, observation):
        mean, logstd = self.parameters
        probs_out = mean(observation)
        sample_ac = probs_out + torch.exp(logstd) * torch.randn(probs_out.size())
        return sample_ac


    def define_train_op(self):
        mean, logstd = self.parameters
        self.optimizer = optim.Adam([logstd] + list(mean.parameters()), lr=self.learning_rate)
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')

    ##################################

    def save(self, filepath):
        mean, logstd = self.parameters
        save_dic = {
                    'logstd': logstd,
                    'mean_preds': mean.state_dict(),
                    }
        torch.save(save_dic, filepath)

    def restore(self, filepath):
        checkpoint = torch.load(filepath)
        mean = MLP(self.ob_dim, output_size=self.ac_dim, n_layers=self.n_layers, size=self.size)
        logstd = checkpoint[checkpoint['logstd']]
        mean.load_state_dict(checkpoint['mean_preds'])
        self.parameters = (mean, logstd)

    ##################################

    # query this policy with observation(s) to get selected action(s)
    @convert_args_to_tensor()
    def get_action(self, obs):
        with torch.no_grad():
            if len(obs.shape)>1:
                observation = obs
            else:
                observation = obs[None]

            # observation = torch.from_numpy(observation).type(torch.FloatTensor)

            # TODO return the action that the policy prescribes
            # HINT1: you will need to call self.sess.run
            # HINT2: the tensor we're interested in evaluating is self.sample_ac
            # HINT3: in order to run self.sample_ac, it will need observation fed into the feed_dict
            return  self._build_action_sampling(observation).numpy()

    # update/train this policy
    def update(self, observations, actions):
        raise NotImplementedError 

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):

    """
        This class is a special case of MLPPolicy,
        which is trained using supervised learning.
        The relevant functions to define are included below.
    """

    @convert_args_to_tensor()
    def update(self, observations, actions):
        assert(self.training, 'Policy must be created with training=True in order to perform training updates...')
        sample_ac = self._build_action_sampling(observations)
        loss = self.mse_criterion(actions, sample_ac)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

