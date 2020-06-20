
import torch

import numpy as np
import torch.nn as nn

from torch import optim
from .base_policy import BasePolicy
from cs285.infrastructure.torch_utils import MLP, multivariate_normal_diag
from torch.distributions import Categorical


class MLPPolicy(BasePolicy):

    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        learning_rate=1e-4,
        training=True,
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
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

        # Build Torch graph
        self.build_graph()

    ##################################

    def build_graph(self):
        # self.define_placeholders()
        self.define_forward_pass_parameters()
        if self.training:
            if self.nn_baseline:
                self.build_baseline_forward_pass_parameters() 
            self.define_train_op()

    ##################################

    # def define_placeholders(self):
    #     raise NotImplementedError

    def define_forward_pass_parameters(self):
        if self.discrete: 
            probs = MLP(self.ob_dim, output_size=self.ac_dim, n_layers=self.n_layers, size=self.size, output_activation=nn.Softmax(dim=1))
            self.parameters = probs
        else:
            mean = MLP(self.ob_dim, output_size=self.ac_dim, n_layers=self.n_layers, size=self.size)
            logstd = torch.zeros(self.ac_dim, requires_grad=True)
            self.parameters = (mean, logstd)

    def _build_action_sampling(self, observation):


        if self.discrete:
            probs = self.parameters
            probs_out = probs(observation)
            sample_ac = torch.squeeze(torch.multinomial(probs_out, num_samples=1), dim=1) # BUG maybe bug happens here
        else:
            mean, logstd = self.parameters
            probs_out = mean(observation)
            sample_ac = probs_out + torch.exp(logstd) * torch.randn(probs_out.size()) # BUG
        return sample_ac

        
    def define_train_op(self):
        if self.discrete:
            probs = self.parameters
            self.optimizer = optim.Adam(probs.parameters(), lr=self.learning_rate)
        else:
            mean, logstd = self.parameters
            self.optimizer = optim.Adam([logstd] + list(mean.parameters()), lr=self.learning_rate)
        
        if self.nn_baseline:
            self.baseline_optimizer = optim.Adam(self.baseline_prediction.parameters(), lr=self.learning_rate)
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')


    def _define_log_prob(self, observation, action):

        if self.discrete:
            #log probability under a categorical distribution
            probs = self.parameters
            logprob_n = torch.distributions.Categorical(probs=probs(observation)).log_prob(action)
        else:
            #log probability under a multivariate gaussian
            mean, logstd = self.parameters
            logprob_n = multivariate_normal_diag(
                loc=mean(observation), scale_diag=torch.exp(logstd)).log_prob(action)
        return logprob_n

    def build_baseline_forward_pass_parameters(self):    
        self.baseline_prediction = MLP(self.ob_dim, output_size=1, n_layers=self.n_layers, size=self.size)

    ##################################

    def save(self, filepath):
        if self.discrete:
            probs = self.parameters
            save_dic = {'probs': probs,}
        else:
            mean, logstd = self.parameters
            save_dic = {
                        'logstd': logstd,
                        'mean_preds': mean.state_dict(),
                        }
        if self.nn_baseline:
            save_dic['nn_baseline'] = self.baseline_prediction.state_dict()
        torch.save(save_dic, filepath)

    def restore(self, filepath):
        checkpoint = torch.load(filepath)
        if self.discrete:
            probs = MLP(self.ob_dim, output_size=self.ac_dim, n_layers=self.n_layers, size=self.size, output_activation=nn.Softmax(dim=-1))
            probs.load_state_dict(checkpoint['probs'])
            self.parameters = probs
        else:
            mean = MLP(self.ob_dim, output_size=self.ac_dim, n_layers=self.n_layers, size=self.size)
            logstd = checkpoint[checkpoint['logstd']]
            mean.load_state_dict(checkpoint['mean_preds'])
            self.parameters = (mean, logstd)
        if self.nn_baseline:
            self.baseline_prediction.load_state_dict(checkpoint['nn_baseline'])

    ##################################

    # update/train this policy
    def update(self, observations, actions): 
        raise NotImplementedError

    # query the neural net that's our 'policy' function, as defined by an mlp above
    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        
        if len(obs.shape)>1:
            observation = obs
        else:
            observation = obs[None]

        observation = torch.from_numpy(observation).type(torch.FloatTensor)

        # TODO return the action that the policy prescribes
        # HINT1: you will need to call self.sess.run
        # HINT2: the tensor we're interested in evaluating is self.sample_ac
        # HINT3: in order to run self.sample_ac, it will need observation fed into the feed_dict
        return  self._build_action_sampling(observation).detach().numpy()

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):

    """
        This class is a special case of MLPPolicy,
        which is trained using supervised learning.
        The relevant functions to define are included below.
    """

    def update(self, observations, actions):
        assert(self.training, 'Policy must be created with training=True in order to perform training updates...')
        sample_ac = self._build_action_sampling(observations)
        loss = self.mse_criterion(actions, sample_ac)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):


    def run_baseline_prediction(self, obs):
        
        observations = torch.from_numpy(obs).type(torch.FloatTensor)
        # TODO: query the neural net that's our 'baseline' function, as defined by an mlp above
        # HINT1: query it with observation(s) to get the baseline value(s)
        # HINT2: see build_baseline_forward_pass (above) to see the tensor that we're interested in
        # HINT3: this will be very similar to how you implemented get_action (above)
        return self.baseline_prediction(observations).detach().numpy()

    def update(self, observations, acs_na, adv_n=None, acs_labels_na=None, qvals=None):
        assert(self.training, 'Policy must be created with training=True in order to perform training updates...')

        adv_n = torch.from_numpy(adv_n).type(torch.FloatTensor)
        observations = torch.from_numpy(observations).type(torch.FloatTensor)
        acs_na = torch.from_numpy(acs_na).type(torch.FloatTensor)
        qvals = torch.from_numpy(qvals).type(torch.FloatTensor)

        self.optimizer.zero_grad()
        
        logprob_n = self._define_log_prob(observations, acs_na)

        loss = -1 * torch.sum(logprob_n * adv_n)


        loss.backward()
        self.optimizer.step()

        if self.nn_baseline:
            targets_n = (qvals - torch.mean(qvals))/(torch.std(qvals)+1e-8)
            # TODO: update the nn baseline with the targets_n
            # HINT1: run an op that you built in define_train_op
            baseline_loss = self.mse_criterion(self.baseline_prediction(observations), targets_n)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
        return loss

#####################################################
#####################################################
