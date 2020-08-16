import torch
import torch.nn.functional as F

from torch import nn
from .base_critic import BaseCritic
from cs285.infrastructure.dqn_utils import huber_loss
from cs285.infrastructure.torch_utils import torch_one_hot, convert_args_to_tensor

class DQNCritic(BaseCritic):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name'] 
        self.ob_dim = hparams['ob_dim']
        self.device = hparams['device']

        if isinstance(self.ob_dim, int):
            self.input_shape = self.ob_dim
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        self._build(hparams['q_func'])

    def _build(self, q_func):

        #####################
 
        # q values, created with the placeholder that holds CURRENT obs (i.e., t)
        self.q_t_values = q_func(self.input_shape, self.ac_dim).to(self.device)

        #####################

        # target q values, created with the placeholder that holds NEXT obs (i.e., t+1)
        self.q_tp1_values = q_func(self.input_shape, self.ac_dim).to(self.device)

        # train_fn will be called in order to train the critic (by minimizing the TD error)
        self.optimizer = self.optimizer_spec.constructor(self.q_t_values.parameters(), **self.optimizer_spec.kwargs)


    def __calc_target_vals(self, next_ob_no, re_n, terminal_n):

        with torch.no_grad():
            # target q values, created with the placeholder that holds NEXT obs (i.e., t+1)
            q_tp1_values = self.q_tp1_values(next_ob_no)

            if self.double_q: 
                # You must fill this part for Q2 of the Q-learning potion of the homework.
                # In double Q-learning, the best action is selected using the Q-network that
                # is being updated, but the Q-value for this action is obtained from the
                # target Q-network. See page 5 of https://arxiv.org/pdf/1509.06461.pdf for more details.
                q_tp1_values_no = self.q_t_values(next_ob_no)
                argmax_slices = torch.stack([torch.arange(q_tp1_values_no.shape[0]), torch.argmax(q_tp1_values_no, dim=1)]).numpy().tolist()
                q_tp1 = q_tp1_values[argmax_slices]

            else:
                # q values of the next timestep
                q_tp1, _ = torch.max(q_tp1_values, dim=1)


            # TODO calculate the targets for the Bellman error
            # HINT1: as you saw in lecture, this would be:
                #currentReward + self.gamma * qValuesOfNextTimestep * (1 - self.done_mask_ph)
            # HINT2: see above, where q_tp1 is defined as the q values of the next timestep
            # HINT3: see the defined placeholders and look for the one that holds current rewards
            return re_n + self.gamma * q_tp1 * (1 - terminal_n)

    def update_target_network(self):
        self.q_tp1_values.load_state_dict(self.q_t_values.state_dict())

    @convert_args_to_tensor()
    def update(self, ob_no, next_ob_no, act_t_ph, re_n, terminal_n, lr):
        ob_no, next_ob_no, act_t_ph, re_n, terminal_n = \
            [x.to(self.device) for x in [ob_no, next_ob_no, act_t_ph, re_n, terminal_n]]
        
        # setting the learning-rate value
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


        target_q_t = self.__calc_target_vals(next_ob_no, re_n, terminal_n)

        q_t = torch.sum(self.q_t_values(ob_no) * torch_one_hot(act_t_ph, self.ac_dim), dim=1)

        ##################### 

        # TODO compute the Bellman error (i.e. TD error between q_t and target_q_t)
        # Note that this scalar-valued tensor later gets passed into the optimizer, to be minimized
        # HINT: use reduce mean of huber_loss (from infrastructure/dqn_utils.py) instead of squared error
        total_error = torch.mean(huber_loss(q_t, target_q_t)) 

        #####################
        
        # train_fn will be called in order to train the critic (by minimizing the TD error)
        self.optimizer.zero_grad()
        total_error.backward()
        nn.utils.clip_grad_norm_(self.q_t_values.parameters(), self.grad_norm_clipping)

        self.optimizer.step()
        return total_error