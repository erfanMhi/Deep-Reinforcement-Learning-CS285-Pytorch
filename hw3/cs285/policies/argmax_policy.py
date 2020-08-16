import torch
import tensorflow as tf

from cs285.infrastructure.torch_utils import convert_args_to_tensor

class ArgMaxPolicy(object):

    def __init__(self, critic, device='cpu'):
        self.critic = critic
        self.device = device

    @convert_args_to_tensor()
    def get_action(self, obs):

        # TODO: Make use of self.action by passing these input observations into self.critic
        # HINT: you'll want to populate the critic's obs_t_ph placeholder
         
        with torch.no_grad():
            if len(obs.shape) > 1:
                observation = obs.to(self.device)
            else:
                observation = obs[None].to(self.device)
            # TODO: Define what action this policy should return
            # HINT1: the critic's q_t_values indicate the goodness of observations, 
            # so they should be used to decide the action to perform
            return torch.argmax(self.critic.q_t_values(observation), dim=1).cpu().numpy()