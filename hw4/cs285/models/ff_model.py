import torch

from .base_model import BaseModel
from cs285.infrastructure.utils import normalize, unnormalize
from cs285.infrastructure.torch_utils import MLP, multivariate_normal_diag, convert_args_to_tensor

class FFModel(BaseModel):

    def __init__(self, ac_dim, ob_dim, n_layers, size, learning_rate=0.001, scope='dyn_model', device='cpu'):
        super(FFModel, self).__init__()

        # init vars
        
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.scope = scope
        self.device = device

        # build graph
        self.build_graph()

    #############################

    def build_graph(self):
        self.define_forward_pass()
        self.define_train_op()
    
    def define_forward_pass(self):
        
       # Hint: Note that the prefix delta is used in the variable below to denote changes in state, i.e. (s'-s)
        self.delta_pred_normalized = MLP(self.ob_dim + self.ac_dim, output_size=self.ob_dim, n_layers=self.n_layers, size=self.size).to(self.device) # TODO(Q1) Use the build_mlp function and the concatenated_input above to define a neural network that predicts unnormalized delta states (i.e. change in state)

    def define_train_op(self):
        
        self.optimizer = torch.optim.Adam(self.delta_pred_normalized.parameters(), lr=self.learning_rate) # TODO(Q1) Define a train_op to minimize the loss defined above. Adam optimizer will work well.

        self.mse_criterion = torch.nn.MSELoss(reduction='mean')


    #############################

    def _forward_delta_pred_normalized(self, observations, actions, data_statistics):
        obs_normalized = normalize(observations, data_statistics['obs_mean'], data_statistics['obs_std']) # TODO(Q1) Define obs_normalized using obs_unnormalized,and self.obs_mean_pl and self.obs_std_pl
        acs_normalized = normalize(actions, data_statistics['acs_mean'], data_statistics['acs_std']) # TODO(Q2) Define acs_normalized using acs_unnormalized and self.acs_mean_pl and self.acs_std_pl
        
        mlp_input = torch.cat([obs_normalized, acs_normalized], dim=1)
        return self.delta_pred_normalized(mlp_input)

    def _get_next_obs_prediction(self, observations, actions, data_statistics):
        
        delta_pred_unnormalized = unnormalize(self._forward_delta_pred_normalized(observations, actions, data_statistics),
                                          data_statistics['delta_mean'], data_statistics['delta_std']) # TODO(Q1) Unnormalize the the delta_pred above using the unnormalize function, and self.delta_mean_pl and self.delta_std_pl

        return observations + delta_pred_unnormalized # TODO(Q1) Predict next observation using current observation and delta prediction (not that next_obs here is unnormalized)


    @convert_args_to_tensor([1, 2], ['obs', 'acs'])
    def get_prediction(self, obs, acs, data_statistics):
        
        with torch.no_grad():
            if len(obs.shape)>1:
                observations = obs.to(self.device)
                actions = acs.to(self.device)
            else:
                observations = obs[None].to(self.device)
                actions = acs[None].to(self.device)

            return self._get_next_obs_prediction(observations, actions, data_statistics).cpu().numpy() # TODO(Q1) Run model prediction on the given batch of data

    @convert_args_to_tensor([1, 2, 3], ['observations', 'actions', 'next_observations'])
    def update(self, observations, actions, next_observations, data_statistics):
        
        observations, actions, next_observations = observations.to(self.device), actions.to(self.device), next_observations.to(self.device)
        # normalize the labels
        delta_labels = next_observations - observations
        delta_labels_normalized = normalize(delta_labels, data_statistics['delta_mean'], data_statistics['delta_std']) # TODO(Q1) Define a normalized version of delta_labels using self.delta_labels (which are unnormalized), and self.delta_mean_pl and self.delta_std_pl

        delta_pred_normalized = self._forward_delta_pred_normalized(observations, actions, data_statistics)
        
        # compared predicted deltas to labels (both should be normalized)
        loss = self.mse_criterion(delta_labels_normalized, delta_pred_normalized) # TODO(Q1) Define a loss function that takes as input normalized versions of predicted change in state and ground truth change in state

        # train the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        
        return loss.detach().cpu()