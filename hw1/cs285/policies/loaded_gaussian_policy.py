import numpy as np
import torch
from .base_policy import BasePolicy
from cs285.infrastructure.torch_utils import LoadedGaussianPolicy, convert_args_to_tensor
import pickle

class Loaded_Gaussian_Policy(BasePolicy):
    def __init__(self, filename, device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.device = device

        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())

        self.nonlin_type = data['nonlin_type']
        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

        assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
        self.policy_params = data[policy_type]

        assert set(self.policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

        # First, observation normalization.
        assert list(self.policy_params['obsnorm'].keys()) == ['Standardizer']
        self.obsnorm_mean = self.policy_params['obsnorm']['Standardizer']['mean_1_D'].astype(np.float32)
        obsnorm_meansq = self.policy_params['obsnorm']['Standardizer']['meansq_1_D'].astype(np.float32)
        self.obsnorm_stdev = torch.from_numpy(np.sqrt(np.maximum(0, obsnorm_meansq - np.square(self.obsnorm_mean))).astype(np.float32)).to(self.device)
        self.obsnorm_mean = torch.from_numpy(self.obsnorm_mean).to(self.device)
        
        self.define_forward_pass()


    ##################################

    def define_forward_pass(self):

        # Build the forward pass
        assert list(self.policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = self.policy_params['hidden']['FeedforwardNet']
        self.gmlp = LoadedGaussianPolicy(layer_params, self.policy_params['out'], self.nonlin_type,
                                    self.obsnorm_mean, self.obsnorm_stdev).to(self.device)
    
    def update(self, obs_no, acs_na, adv_n=None, acs_labels_na=None):
        print("\n\nThis policy class simply loads in a particular type of policy and queries it.")
        print("Not training procedure has been written, so do not try to train it.\n\n")
        raise NotImplementedError
    
    @convert_args_to_tensor()
    def get_action(self, obs):
        with torch.no_grad():
            if len(obs.shape)>1:
                observation = obs.to(self.device)
            else:
                observation = obs[None, :].to(self.device)
            return self.gmlp(observation).cpu().numpy()

