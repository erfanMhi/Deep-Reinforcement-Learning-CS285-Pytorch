import numpy as np 

from collections import OrderedDict

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyAC
from cs285.critics.bootstrapped_continuous_critic import BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *

class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env 
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
                               self.agent_params['ac_dim'],
                               self.agent_params['ob_dim'],
                               self.agent_params['n_layers'],
                               self.agent_params['size'],
                               discrete=self.agent_params['discrete'],
                               learning_rate=self.agent_params['learning_rate'],
                               device = self.agent_params['device']
                               )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
         
        # TODO Implement the following pseudocode:
            # 1) query the critic with ob_no, to get V(s)
            # 2) query the critic with next_ob_no, to get V(s')
            # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
            # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
            # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        adv_n = re_n - self.critic.forward(ob_no) + self.gamma*self.critic.forward(next_ob_no) *  np.logical_not(terminal_n)

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        
        # TODO Implement the following pseudocode:
            # for agent_params['num_critic_updates_per_agent_update'] steps,
            #     update the critic
 
            # advantage = estimate_advantage(...)

            # for agent_params['num_actor_updates_per_agent_update'] steps,
            #     update the actor
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.critic.update(ob_no, next_ob_no, re_n, terminal_n)
        
        advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
 
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss = self.actor.update(ob_no, ac_na, adv_n=advantage)

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss  # put final critic loss here
        loss['Actor_Loss'] = actor_loss  # put final actor loss here
        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)