import torch
import numpy as np
import torch.nn.functional as F

from .base_critic import BaseCritic
from cs285.infrastructure.tf_utils import build_mlp
from cs285.infrastructure.torch_utils import MLP, multivariate_normal_diag, convert_args_to_tensor
from torch import optim


class BootstrappedContinuousCritic(BaseCritic):
    def __init__(self, hparams):
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.device = hparams['device']
        self._build()

    def _build(self):
        """
            Notes on notation:

            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function

            Prefixes and suffixes:
            ob - observation 
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)

            Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
            is None

            ----------------------------------------------------------------------------------
            loss: a function of self.sy_ob_no, self.sy_ac_na and self.sy_adv_n that we will differentiate
                to get the policy gradient.
        """
        # define the critic
        self._critic_prediction = MLP(self.ob_dim, 1, self.n_layers, self.size).to(self.device)
        self.critic_prediction = lambda ob: torch.squeeze(self._critic_prediction(ob))
        # TODO: set up the critic loss
        # HINT1: the critic_prediction should regress onto the targets placeholder (sy_target_n)
        # HINT2: use tf.losses.mean_squared_error
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')

        # TODO: use the AdamOptimizer to optimize the loss defined above
        self.optimizer = optim.Adam(self._critic_prediction.parameters(), lr=self.learning_rate)

    @convert_args_to_tensor()
    def forward(self, ob, get_torch=False):
        # TODO: run your critic
        # HINT: there's a neural network structure defined above with mlp layers, which serves as your 'critic'
        with torch.no_grad():
            if get_torch:
                return self.critic_prediction(ob)
            
            ob = ob.to(self.device)
            return self.critic_prediction(ob).cpu().numpy()

    @convert_args_to_tensor()
    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the sampled paths
            let num_paths be the number of sampled paths

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                loss
        """
        ob_no, next_ob_no, re_n, terminal_n = \
            [x.to(self.device) for x in [ob_no, next_ob_no, re_n, terminal_n]]
        # TODO: Implement the pseudocode below: 

        # do the following (self.num_grad_steps_per_target_update * self.num_target_updates) times:
            # every self.num_grad_steps_per_target_update steps (which includes the first step),
                # recompute the target values by 
                    #a) calculating V(s') by querying this critic network (ie calling 'forward') with next_ob_no
                    #b) and computing the target values as r(s, a) + gamma * V(s')
                # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it to 0) when a terminal state is reached
            # every time,
                # update this critic using the observations and targets
                # HINT1: need to sess.run the following: 
                    #a) critic_update_op 
                    #b) critic_loss
                # HINT2: need to populate the following (in the feed_dict): 
                    #a) sy_ob_no with ob_no
                    #b) sy_target_n with target values calculated above
        # print(next_ob_no)
        for _ in range(self.num_target_updates):
            target_values = re_n + self.gamma*self.forward(next_ob_no, get_torch=True) * torch.logical_not(terminal_n)
            for _ in range(self.num_grad_steps_per_target_update):
                self.optimizer.zero_grad()

                loss = self.mse_criterion(self.critic_prediction(ob_no), target_values)

                loss.backward()
                self.optimizer.step()

        return loss.item()