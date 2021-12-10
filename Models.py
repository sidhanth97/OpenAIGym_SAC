import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import OrderedDict
import numpy as np


def init_weights(layer):
    """
    Initializing weight matrices based on Kaiming Uniform
    :param layer: Pytorch layer
    :return: None
    """
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, a=0.2)
        layer.bias.data.fill_(0.00001)


class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, name, device,
                 model_dir, fc1_dims=256, fc2_dims=256):
        """
        Constructor for actor network
        :param lr: Learning rate for actor
        :param input_dims: Input dimension (Latent vector size)
        :param n_actions: Number of actions
        :param name: Name of actor model
        :param device: CUDA or CPU
        :param model_dir: Model directory for actor
        :param fc1_dims: Layer-1 dimensions
        :param fc2_dims: Layer-2 dimensions
        """
        super(ActorNetwork, self).__init__()
        self.policy_network = nn.Sequential(OrderedDict([
            ('Linear_1', nn.Linear(*input_dims, fc1_dims)),
            ('LeakyReLU_1', nn.LeakyReLU(0.2)),
            ('Linear_2', nn.Linear(fc1_dims, fc2_dims)),
            ('LeakyReLU_2', nn.LeakyReLU(0.2)),
        ]))

        self.mean = nn.Linear(fc2_dims, n_actions)
        self.logvar = nn.Linear(fc2_dims, n_actions)
        self.apply(init_weights)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(self.device)
        self.checkpoint_file = model_dir + '/' + name + '.pth'
        self.reparam_noise = 1e-6

    def gaussian_policy(self, state, reparameterize=True):
        """
        Function to sample action from a gaussian policy
        :param state: State tensor
        :param reparameterize: Set to True while training else False
        :return: action, log_prob
        """
        # Sampling actions from gaussian
        mu, logstd = self.forward(state)
        pdf = Normal(mu, logstd.exp())
        actions = pdf.rsample() if reparameterize else pdf.sample()
        batch_size = actions.shape[0]

        # Computing Log-Prob of actions
        # Refer line 53-58 from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
        log_prob_correction = (2 * (np.log(2) - actions - F.softplus(-2*actions))).sum(axis=1)
        log_prob = pdf.log_prob(actions).sum(axis=-1) - log_prob_correction

        # Setting actions to desired output-range
        # Gas and Brake actions aren't independent and has to handled differently (Dramatic improvement in training)
        # Brake is set to 0.1 if the tendency to accelerate is higher than to brake
        action = T.tanh(actions)
        steering_action = action[:, 0].view(batch_size, 1)
        gas_action = ((action[:, 1] + 1) / 2).view(batch_size, 1)
        brake_action = 0.175 * ((action[:, 2] + 1) / 2).view(batch_size, 1)
        brake_action[gas_action > brake_action] = 0.1
        action = T.cat((steering_action, gas_action, brake_action), dim=1)
        return action, log_prob

    def forward(self, state):
        """
        Forward pass function on the actor network
        :param state: state tensor
        :return: mean and log-std of the gaussian that the action should be sampled from
        """
        # Log-std can afford to be negative whereas, std should only be positive
        # Log-std is clamped between [-20, 2] as mentioned in the paper (mostly for stability)
        pdf = self.policy_network(state)
        mu = self.mean(pdf)
        logstd = self.logvar(pdf)
        logstd = T.clamp(logstd, min=-20, max=2)
        return mu, logstd

    def save_checkpoint(self):
        """
        Save function for actor network
        :return: None
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load function for actor network
        :return: None
        """
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, name, device,
                 model_dir, fc1_dims=256, fc2_dims=256):
        """
        Constructor for critic network
        :param lr: Learning rate for critic network
        :param input_dims: Input dimension for critic network state-action value network
        :param n_actions: Number of actions
        :param name: Name of critic model
        :param device: CUDA or CPU
        :param model_dir: Model directory for critic
        :param fc1_dims: Layer-1 dimensions
        :param fc2_dims: Layer-2 dimensions
        """
        # Two networks to implement the Double-Q trick
        super(CriticNetwork, self).__init__()
        self.state_action_network_1 = nn.Sequential(OrderedDict([
            ('Linear_1', nn.Linear(input_dims[0] + n_actions, fc1_dims)),
            ('LeakyReLU_1', nn.LeakyReLU(0.2)),
            ('Linear_2', nn.Linear(fc1_dims, fc2_dims)),
            ('LeakyReLU_2', nn.LeakyReLU(0.2)),
            ('Final_layer', nn.Linear(fc2_dims, 1))
        ]))

        self.state_action_network_2 = nn.Sequential(OrderedDict([
            ('Linear_1', nn.Linear(input_dims[0] + n_actions, fc1_dims)),
            ('LeakyReLU_1', nn.LeakyReLU(0.2)),
            ('Linear_2', nn.Linear(fc1_dims, fc2_dims)),
            ('LeakyReLU_2', nn.LeakyReLU(0.2)),
            ('Final_layer', nn.Linear(fc2_dims, 1))
        ]))

        self.apply(init_weights)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(self.device)
        self.checkpoint_file = model_dir + '/' + name + '.pth'

    def forward(self, state, action):
        """
        Forward pass function for the critic network
        :param state: state tensor
        :param action: action tensor
        :return: Q-values from the two Q networks
        """
        # SAC uses clipped double-Q trick and the estimated Q-value is minimum of the two Q-values
        q_value_1 = self.state_action_network_1(T.cat([state, action], dim=1))
        q_value_2 = self.state_action_network_2(T.cat([state, action], dim=1))
        return q_value_1, q_value_2

    def save_checkpoint(self):
        """
        Save function for critic network
        :return: None
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load function for actor network
        :return: None
        """
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
