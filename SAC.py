import numpy as np
import torch as T
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer
from Models import ActorNetwork, CriticNetwork


class SACAgent:
    def __init__(self, device, model_dir, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 input_dims=[32], n_actions=3, init_temperature=0.2, gamma=0.99,
                 actor_frequency=1, target_critic_frequency=10, tau=0.05,
                 batch_size=256, max_size=1000000):
        """
        Constructor for SAC Agent class
        :param device: CUDA or CPU
        :param model_dir: Agent model directory to save
        :param lr_actor: Learning rate for actor
        :param lr_critic: Learning rate for critic
        :param lr_alpha: Learning rate for temperature
        :param input_dims: Input dimension to the SAC model, in this case it's the size of the latent vector from VAE
        :param n_actions: Number of possible actions -> Steering, Gas and Brake
        :param init_temperature: Initial temperature value
        :param gamma: Discount factor
        :param actor_frequency: Update frequency for actor
        :param target_critic_frequency: Update frequency for target-critic
        :param tau: Soft update Polyak averaging parameter
        :param batch_size: Batch size of MDP to train
        :param max_size: Maximum size of replay buffer
        """
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.actor_frequency = actor_frequency
        self.target_critic_frequency = target_critic_frequency
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.actor = ActorNetwork(lr_actor, input_dims, n_actions,
                                  'ActorNetwork', device, model_dir)
        self.critic = CriticNetwork(lr_critic, input_dims, n_actions,
                                    'CriticNetwork', device, model_dir)
        self.target_critic = CriticNetwork(lr_critic, input_dims, n_actions,
                                           'TargetCriticNetwork', device, model_dir)
        self.soft_update_critic(tau=1)
        self.entropy = T.tensor(np.log(init_temperature), requires_grad=True, device=device)
        self.target_entropy = -1 * n_actions
        self.entropy_optim = T.optim.Adam([self.entropy], lr=lr_alpha)

    def choose_action(self, state, reparameterize=True):
        """
        Agent chooses action from the gaussian policy
        :param state: State tensor
        :param reparameterize: Flag, set to True while training else False
        :return: action
        """
        action, _ = self.actor.gaussian_policy(state, reparameterize=reparameterize)
        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        """
        Remember a Markov decision process / trajectory given by the environment
        :param state: State tensor
        :param action: Action Tensor
        :param reward: Reward Tensor
        :param new_state: New State tensor
        :param done: Terminal State tensor
        :return: None
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def soft_update_critic(self, tau=None):
        """
        Soft updates the target critic based on Polyak averaging
        :param tau: Polyak averaging parameter
        :return: None
        """
        if tau is None:
            tau = self.tau

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()
        target_critic_dict = dict(target_critic_params)
        critic_dict = dict(critic_params)
        for name in critic_dict:
            critic_dict[name] = tau * critic_dict[name].clone() + \
                                (1 - tau) * target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_dict)

    def learn(self, iteration):
        """
        Off policy SAC agent learning update
        :param iteration: Iteration number
        :return: None
        """
        if self.memory.mem_counter < self.batch_size:
            return

        # Sample from Replay Buffer
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        reward = T.tensor(reward, dtype=T.float).to(self.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.device)
        not_done = T.tensor(~done).to(self.device)

        # Update Critic
        with T.no_grad():
            next_action, next_log_prob = self.actor.gaussian_policy(new_state)
            target_q1, target_q2 = self.target_critic.forward(new_state, next_action)
            target_value = T.min(target_q1, target_q2) - (self.entropy.exp() * next_log_prob).view(-1, 1)
            target_q = (reward.view(-1, 1) + self.gamma * not_done.view(-1, 1) * target_value).detach()

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Actor needs to update lower than critic
        if iteration % self.actor_frequency == 0:
            # Setting gradient computation to False for critic
            for params in self.critic.parameters():
                params.requires_grad = False

            # Update Actor
            action, log_prob = self.actor.gaussian_policy(state)
            q1, q2 = self.critic.forward(state, action)
            actor_loss = (self.entropy.exp() * log_prob - T.min(q1, q2)).mean()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update Temperature
            entropy_difference = (-1. * log_prob - self.target_entropy).detach()
            entropy_loss = (self.entropy.exp() * entropy_difference).mean()
            self.entropy_optim.zero_grad()
            entropy_loss.backward()
            self.entropy_optim.step()

            # Setting gradient computation to True for critic
            for params in self.critic.parameters():
                params.requires_grad = True

        # Soft update target-critic
        with T.no_grad():
            if iteration % self.target_critic_frequency == 0:
                self.soft_update_critic(self.tau)

        return critic_loss.item(), actor_loss.item(), entropy_loss.item()

    def save_model(self):
        """
        Helper function to save the models
        :return: None
        """
        print('Saving Models')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_model(self):
        """
        Helper function to load the models
        :return: None
        """
        print('Loading Models')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
