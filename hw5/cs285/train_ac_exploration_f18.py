"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Soroush Nasiriany, Sid Reddy, and Greg Kahn
Adapted for CS294-112 Fall 2018 with <3 by Michael Chang, some experiments by Greg Kahn, beta-tested by Sid Reddy
"""
import numpy as np
import torch
import torch.nn as nn
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process

from exploration import ExemplarExploration, DiscreteExploration, RBFExploration
from density_model import Exemplar, Histogram, RBF
from utils.pytorch_utils import *

#============================================================================================#
# Utilities
#============================================================================================#

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_AC)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

#============================================================================================#
# Actor Critic
#============================================================================================#

class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_advantage_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']
        self.num_target_updates = computation_graph_args['num_target_updates']
        self.num_grad_steps_per_target_update = computation_graph_args['num_grad_steps_per_target_update']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_advantage_args['gamma']
        self.normalize_advantages = estimate_advantage_args['normalize_advantages']

        self.device = computation_graph_args['device']
        self._build_graph()
    
    def _build_graph(self):
        self._build_actor()
        self._build_critic()

    ##################### Building Actor #######################
    def _policy_forward_pass(self):
        """ Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)

            arguments:
                sy_ob_no: (batch_size, self.ob_dim)

            returns:
                the parameters of the policy.

                if discrete, the parameters are the logits of a categorical distribution
                    over the actions
                    sy_probs_na: (batch_size, self.ac_dim)

                if continuous, the parameters are a tuple (mean, log_std) of a Gaussian
                    distribution over actions. log_std should just be a trainable
                    variable, not a network output.
                    sy_mean: (batch_size, self.ac_dim)
                    sy_logstd: (self.ac_dim,)

            Hint: use the 'build_mlp' function to output the logits (in the discrete case)
                and the mean (in the continuous case).
                Pass in self.n_layers for the 'n_layers' argument, and
                pass in self.size for the 'size' argument.
        """
        if self.discrete:
            sy_probs_na = MLP(self.ob_dim, output_size=self.ac_dim, n_layers=self.n_layers, size=self.size, output_activation=nn.Softmax(dim=1))
            self.policy_parameters = sy_probs_na.to(self.device)
        else:
            sy_mean = MLP(self.ob_dim, output_size=self.ac_dim, n_layers=self.n_layers, size=self.size)
            sy_logstd = torch.zeros(self.ac_dim, requires_grad=True)
            self.policy_parameters = (sy_mean.to(self.device), sy_logstd.to(self.device))

        
    def _define_actor_train_op(self):
        
        if self.discrete:
            probs = self.policy_parameters
            self.actor_optimizer = torch.optim.Adam(probs.parameters(), lr=self.learning_rate)
        else:
            mean, logstd = self.policy_parameters
            self.actor_optimizer = torch.optim.Adam([logstd] + list(mean.parameters()), lr=self.learning_rate)
              
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')

    def _build_actor(self):

        # defining forward pass of the policy
        self.policy_forward_pass()

        # define operation that are needed for backpropagation
        self._define_actor_train_op()

    def _define_log_prob(self, observation, action):
            sy_logits_na = build_mlp(sy_ob_no, self.ac_dim, 'discrete_logits', n_layers=self.n_layers, size=self.size)
            return sy_logits_na
        else:
            sy_mean = build_mlp(sy_ob_no, self.ac_dim, 'continuous_logits', n_layers=self.n_layers, size=self.size)
            sy_logstd = tf.Variable(tf.zeros(self.ac_dim), name='sy_logstd')
            return (sy_mean, sy_logstd)

    def sample_action(self, policy_parameters):
        """ Constructs a symbolic operation for stochastically sampling from the policy
            distribution

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

            returns:
                sy_sampled_ac: 
                    if discrete: (batch_size)
                    if continuous: (batch_size, self.ac_dim)

            Hint: for the continuous case, use the reparameterization trick:
                 The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
        
                      mu + sigma * z,         z ~ N(0, I)
        
                 This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
        """
        if self.discrete:
            sy_logits_na = policy_parameters
            sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, num_samples=1), axis=1)
        else:
            sy_mean, sy_logstd = policy_parameters
            sy_sampled_ac = sy_mean + tf.exp(sy_logstd) * tf.random_normal(tf.shape(sy_mean), 0, 1)
        return sy_sampled_ac

    def get_log_prob(self, policy_parameters, sy_ac_na):
        """ Constructs a symbolic operation for computing the log probability of a set of actions
            that were actually taken according to the policy

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_probs_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

                sy_ac_na: (batch_size, self.ac_dim)

            returns:
                sy_logprob_n: (batch_size)

            Hint:
                For the discrete case, use the log probability under a categorical distribution.
                For the continuous case, use the log probability under a multivariate gaussian.
        """
 
        if self.discrete:
            #log probability under a categorical distribution
            sy_probs_na = self.policy_parameters
            sy_logprob_n = torch.distributions.Categorical(probs=sy_probs_na(observation)).log_prob(action)
        else:
            #log probability under a multivariate gaussian
            sy_mean, sy_logstd = self.policy_parameters
            sy_logprob_n = multivariate_normal_diag(
                loc=sy_mean(observation), scale_diag=torch.exp(sy_logstd)).log_prob(action)
        return sy_logprob_n
    
    ############################### Building Critic ##############################
    
    def _build_critic(self):
        # define the critic
        self._critic_prediction = MLP(self.ob_dim, 1, self.n_layers, self.size).to(self.device)
        self.critic_prediction = lambda ob: torch.squeeze(self._critic_prediction(ob))

        # use the AdamOptimizer to optimize the loss defined above
        self.critic_optimizer = torch.optim.Adam(self._critic_prediction.parameters(), lr=self.learning_rate)
   
    def _sample_action(self, observation):
        """ Constructs a symbolic operation for stochastically sampling from the policy
            distribution

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_probs_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

            returns:
                sy_sampled_ac: 
                    if discrete: (batch_size)
                    if continuous: (batch_size, self.ac_dim)

            Hint: for the continuous case, use the reparameterization trick:
                 The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
        
                      mu + sigma * z,         z ~ N(0, I)
        
                 This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
        """
        if self.discrete:
            sy_probs_na = self.policy_parameters
            sample_ac = torch.squeeze(torch.multinomial(sy_probs_na(observation), num_samples=1), dim=1) # BUG maybe bug happens here
        else:
            sy_mean, sy_logstd = self.policy_parameters
            probs_out = sy_mean(observation)
            sample_ac = probs_out + torch.exp(sy_logstd) * torch.randn(probs_out.size(), device=self.device) # BUG
        return sample_ac


    @convert_args_to_tensor()
    def get_action(self, obs):
        with torch.no_grad():
            if len(obs.shape)>1:
                observation = obs.to(self.device)
            else:
                observation = obs[None].to(self.device)

            # observation = torch.from_numpy(observation).type(torch.FloatTensor)

            # TODO return the action that the policy prescribes
            # HINT1: you will need to call self.sess.run
            # HINT2: the tensor we're interested in evaluating is self.sample_ac
            # HINT3: in order to run self.sample_ac, it will need observation fed into the feed_dict
            return  self._sample_action(observation).cpu().numpy()

    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards, next_obs, terminals = [], [], [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            ac = self.get_action(ob)
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            next_obs.append(ob)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        path = {"observation" : np.array(obs, dtype=np.float32), 
                "reward" : np.array(rewards, dtype=np.float32), 
                "action" : np.array(acs, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "terminal": np.array(terminals, dtype=np.float32)}
        return path

    @convert_args_to_tensor()
    def critic_forward(self, ob, get_torch=False):
        # run your critic
        with torch.no_grad():
            if get_torch:
                return self.critic_prediction(ob)
            
            ob = ob.to(self.device)
            return self.critic_prediction(ob).cpu().numpy()

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Estimates the advantage function value for each timestep.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths
        """
        
        adv_n = re_n - self.critic_forward(ob_no) + self.gamma*self.critic_forward(next_ob_no) *  np.logical_not(terminal_n)
        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    @convert_args_to_tensor()
    def update_critic(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                nothing
        """
        ob_no, next_ob_no, re_n, terminal_n = \
            [x.to(self.device) for x in [ob_no, next_ob_no, re_n, terminal_n]]

        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if i % self.num_grad_steps_per_target_update == 0:
                target_values = re_n + self.gamma*self.critic_forward(next_ob_no, get_torch=True) * torch.logical_not(terminal_n)
            
            self.critic_optimizer.zero_grad()

            loss = self.mse_criterion(self.critic_prediction(ob_no), target_values)

            loss.backward()
            self.critic_optimizer.step()

    @convert_args_to_tensor()
    def update_actor(self, ob_no, ac_na, adv_n):
        """ 
            Update the parameters of the policy.

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths).
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths

            returns:
                nothing

        """
    
        observations, acs_na, adv_n = \
            [x.to(self.device) if x is not None else None for x in [ob_no, ac_na, adv_n]] 

        self.actor_optimizer.zero_grad()
        
        logprob_n = self._define_log_prob(observations, acs_na)

        loss = -1 * torch.sum(logprob_n * adv_n)


        loss.backward()
        self.actor_optimizer.step()

def train_AC(
        exp_name,
        env_name,
        n_iter, 
        gamma, 
        min_timesteps_per_batch, 
        max_path_length,
        learning_rate,
        num_target_updates,
        num_grad_steps_per_target_update,
        animate, 
        logdir, 
        normalize_advantages,
        seed,
        n_layers,
        size,
        ########################################################################
        # Exploration args
        bonus_coeff,
        kl_weight,
        density_lr,
        density_train_iters,
        density_batch_size,
        density_hiddim,
        dm,
        replay_size,
        sigma,
        ########################################################################
        ):
    start = time.time()

    #========================================================================================#
    # Set Up Logger
    #========================================================================================#
    setup_logger(logdir, locals())

    #========================================================================================#
    # Set Up Env
    #========================================================================================#

    # Make the gym environment
    ########################################################################
    # Exploration
    if env_name == 'PointMass-v0':
        from pointmass import PointMass
        env = PointMass()
    else:
        env = gym.make(env_name)
    dirname = logz.G.output_dir
    ########################################################################

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Is this env continuous or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    # Initialize Agent
    #========================================================================================#
    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
        'num_target_updates': num_target_updates,
        'num_grad_steps_per_target_update': num_grad_steps_per_target_update,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_advantage_args = {
        'gamma': gamma,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_advantage_args) #estimate_return_args

    # build computation graph
    agent.build_computation_graph()

    ########################################################################
    # Initalize exploration density model
    if dm != 'none':
        if env_name == 'PointMass-v0' and dm == 'hist':
            density_model = Histogram(
                nbins=env.grid_size, 
                preprocessor=env.preprocess)
            exploration = DiscreteExploration(
                density_model=density_model,
                bonus_coeff=bonus_coeff)
        elif dm == 'rbf':
            density_model = RBF(sigma=sigma)
            exploration = RBFExploration(
                density_model=density_model,
                bonus_coeff=bonus_coeff,
                replay_size=int(replay_size))
        elif dm == 'ex2':
            density_model = Exemplar(
                ob_dim=ob_dim, 
                hid_dim=density_hiddim,
                learning_rate=density_lr, 
                kl_weight=kl_weight)
            exploration = ExemplarExploration(
                density_model=density_model, 
                bonus_coeff=bonus_coeff, 
                train_iters=density_train_iters, 
                bsize=density_batch_size,
                replay_size=int(replay_size))
            exploration.density_model.build_computation_graph()
        else:
            raise NotImplementedError

    ########################################################################

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    ########################################################################
    if dm != 'none':
        exploration.receive_tf_sess(agent.sess)
    ########################################################################

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = np.concatenate([path["reward"] for path in paths])
        next_ob_no = np.concatenate([path["next_observation"] for path in paths])
        terminal_n = np.concatenate([path["terminal"] for path in paths])

        ########################################################################
        # Modify the reward to include exploration bonus
        """
            1. Fit density model
                if dm == 'ex2':
                    the call to exploration.fit_density_model should return ll, kl, elbo
                else:
                    the call to exploration.fit_density_model should return nothing
            2. Modify the re_n with the reward bonus by calling exploration.modify_reward
        """
        old_re_n = re_n
        if dm == 'none':
            pass
        else:
            # 1. Fit density model
            if dm == 'ex2':
                ### PROBLEM 3
                ### YOUR CODE HERE
                ll, kl, elbo = exploration.fit_density_model(ob_no)
            elif dm == 'hist' or dm == 'rbf':
                ### PROBLEM 1
                ### YOUR CODE HERE
                exploration.fit_density_model(ob_no)
            else:
                assert False

            # 2. Modify the reward
            ### PROBLEM 1
            ### YOUR CODE HERE
            re_n = exploration.modify_reward(re_n, ob_no)
            
            print('average state', np.mean(ob_no, axis=0))
            print('average action', np.mean(ac_na, axis=0))

            # Logging stuff.
            # Only works for point mass.
            if env_name == 'PointMass-v0':
                np.save(os.path.join(dirname, '{}'.format(itr)), ob_no)
        ########################################################################
        agent.update_critic(ob_no, next_ob_no, re_n, terminal_n)
        adv_n = agent.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        agent.update_actor(ob_no, ac_na, adv_n)

        if n_iter - itr < 10:
            max_reward_path_idx = np.argmax(np.array([path["reward"].sum() for path in paths]))
            print(paths[max_reward_path_idx]['reward'])

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        ########################################################################
        logz.log_tabular("Unmodified Rewards Mean", np.mean(old_re_n))
        logz.log_tabular("Unmodified Rewards Std", np.mean(old_re_n))
        logz.log_tabular("Modified Rewards Mean", np.mean(re_n))
        logz.log_tabular("Modified Rewards Std", np.mean(re_n))
        if dm == 'ex2':
            logz.log_tabular("Log Likelihood Mean", np.mean(ll))
            logz.log_tabular("Log Likelihood Std", np.std(ll))
            logz.log_tabular("KL Divergence Mean", np.mean(kl))
            logz.log_tabular("KL Divergence Std", np.std(kl))
            logz.log_tabular("Negative ELBo", -elbo)
        ########################################################################
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()

def main():
    from gym.envs.registration import register
    register(
        id='sparse-cheetah-cs285-v1',
        entry_point='cs285.sparse_half_cheetah:HalfCheetahEnv',
        max_episode_steps=1000,
    )
    from cs285.sparse_half_cheetah import HalfCheetahEnv

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vac')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=32)
    ########################################################################
    parser.add_argument('--bonus_coeff', '-bc', type=float, default=1e-3)
    parser.add_argument('--density_model', type=str, default='hist | rbf | ex2 | none')
    parser.add_argument('--kl_weight', '-kl', type=float, default=1e-2)
    parser.add_argument('--density_lr', '-dlr', type=float, default=5e-3)
    parser.add_argument('--density_train_iters', '-dti', type=int, default=1000)
    parser.add_argument('--density_batch_size', '-db', type=int, default=64)
    parser.add_argument('--density_hiddim', '-dh', type=int, default=32)
    parser.add_argument('--replay_size', '-rs', type=int, default=int(1e6))
    parser.add_argument('--sigma', '-sig', type=float, default=0.2)
    ########################################################################

    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = 'ac_' + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_AC(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                num_target_updates=args.num_target_updates,
                num_grad_steps_per_target_update=args.num_grad_steps_per_target_update,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                ########################################################################
                bonus_coeff=args.bonus_coeff,
                kl_weight=args.kl_weight,
                density_lr=args.density_lr,
                density_train_iters=args.density_train_iters,
                density_batch_size=args.density_batch_size,
                density_hiddim=args.density_hiddim,
                dm=args.density_model,
                replay_size=args.replay_size,
                sigma=args.sigma
                ########################################################################
                )

        if args.n_experiments > 1:
            # # Awkward hacky process runs, because Tensorflow does not like
            # # repeatedly calling train_AC in the same thread.
            p = Process(target=train_func, args=tuple())
            p.start()
            processes.append(p)
            # if you comment in the line below, then the loop will block
            # until this process finishes
            # p.join()
        else:
            train_func()

    for p in processes:
        p.join()
        

if __name__ == "__main__":
    main()
