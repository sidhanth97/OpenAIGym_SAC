import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        """
        Constructor class for replay/sample buffer
        :param max_size: Maximum memory size
        :param input_shape: Input shape of the state (Latent vector size from VAE)
        :param n_actions: Number of actions
        """
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        """
        Function to store a memory transition to enable off-policy learning
        :param state: State
        :param action: Action
        :param reward: Instataneous reward
        :param new_state: Next state
        :param done: Done flag
        :return: None
        """
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        """
        Function to sample memory from replay/sample buffer
        :param batch_size: Batch of samples to sample
        :return: MDP variables
        """
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        done_batch = self.terminal_memory[batch]
        return state_batch, action_batch, reward_batch, new_state_batch, done_batch


