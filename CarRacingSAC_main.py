import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as t
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import os
import gym
from gym.envs.box2d import car_racing
from gym.envs.box2d.car_dynamics import Car
import numpy as np
from PIL import Image as Image
import cv2
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from SAC import SACAgent
from VAE import VAE


def convert_observation(obs):
    state = np.flip(np.expand_dims(obs, axis=0), axis=0).copy()
    state = T.einsum("nhwc -> nchw", T.from_numpy(state)).type('torch.FloatTensor')
    return state


def generate_action(prev_action):
    if np.random.randint(3) % 3:
        return prev_action
    index = np.random.randn(3)
    index[1] = np.abs(index[1])
    index = np.argmax(index)
    mask = np.zeros(3)
    mask[index] = 1
    action = np.random.randn(3)
    action = np.tanh(action)
    action[1] = (action[1] + 1) / 2
    action[2] = (action[2] + 1) / 2
    return action * mask


def get_video(test, pred, agent_name):
    size = (192, 85)
    video_fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    video = cv2.VideoWriter('videos/' + agent_name + '.mp4', video_fourcc, 60, size)
    print('Converting predictions to video')
    for i in tqdm(range(test.shape[0])):
        arr1 = np.einsum("hwc -> hwc", test[i, :, :, :])
        arr2 = np.einsum("chw -> hwc", pred[i, :, :, :])
        arr1 = np.asarray(arr1[:85, :, :], dtype=np.uint8)
        arr2 = np.asarray(arr2[:85, :, :], dtype=np.uint8)
        arr = cv2.hconcat([arr1, arr2])
        video.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    video.release()


def train_agent(load_model_flag, episodes=1000, n_steps=5000000, start_policy=150,
                eval_interval=100, update=100, random_exploration=True, batch_size=256):
    total_counter = 0
    learning_updates = 0
    prev_avg_reward = -np.inf
    for episode in range(episodes):
        score = 0
        episode_counter = 0
        done = False
        observation = env.reset()
        state = convert_observation(observation)
        cuda_state = state.to(device)
        z = VAE_network.forward(cuda_state, encoder=True, decoder=False)
        position = np.random.randint(len(env.track))
        env.car = Car(env.world, *env.track[position][1:4])
        if random_exploration:
            action = env.action_space.sample()

        while not done:
            if episode < start_policy and not load_model_flag:
                action = generate_action(action) if random_exploration else env.action_space.sample()
            else:
                action = agent.choose_action(z)

            new_observation, reward, done, info = env.step(action)
            new_state = convert_observation(new_observation)
            cuda_new_state = new_state.to(device)
            new_z = VAE_network.forward(cuda_new_state, encoder=True, decoder=False)
            if episode_counter > batch_size and episode_counter % update == 0:
                for j in range(update):
                    c_loss, a_loss, e_loss = agent.learn(iteration=learning_updates)
                    writer.add_scalar('Critic_loss', c_loss, learning_updates)
                    writer.add_scalar('Actor_loss', a_loss, learning_updates)
                    writer.add_scalar('Entropy_loss', e_loss, learning_updates)
                    learning_updates += 1

            done = False if episode_counter == env._max_episode_steps else done
            agent.remember(z.cpu().detach().numpy(), action, reward, new_z.cpu().detach().numpy(), done)
            z = new_z
            episode_counter += 1
            total_counter += 1
            score += reward
            if episode_counter > n_steps / 10 or episode_counter == env._max_episode_steps:
                observation = env.reset()
                state = convert_observation(observation)
                z = VAE_network.forward(state.to(device), encoder=True, decoder=False)
                done = True

            env.render()

        writer.add_scalar('Return', score, episode)
        if total_counter > n_steps:
            print('Done with Training')
            break

        if (episode + 1) % eval_interval == 0:
            episodic_rewards = []
            eval_episodes = 10
            for _ in range(eval_episodes):
                observation = env.reset()
                state = convert_observation(observation)
                cuda_state = state.to(device)
                z = VAE_network.forward(cuda_state, encoder=True, decoder=False)
                done = False
                episode_reward = 0
                while not done:
                    action = agent.choose_action(z, reparameterize=False)
                    new_observation, reward, done, info = env.step(action)
                    new_state = convert_observation(new_observation)
                    cuda_new_state = new_state.to(device)
                    new_z = VAE_network.forward(cuda_new_state, encoder=True, decoder=False)
                    z = new_z
                    episode_reward += reward
                    if episode_reward < -500:
                        print('Policy needs to train')
                        load_model_flag = False
                        break
                    env.render()

                episodic_rewards.append(episode_reward)

            curr_avg_reward = np.mean(episodic_rewards)
            writer.add_scalar('Evaluated-Avg-Return', curr_avg_reward, int((episode + 1) / eval_interval))
            if curr_avg_reward > prev_avg_reward:
                load_model_flag = True
                agent.save_model()
                prev_avg_reward = curr_avg_reward

            print('Return: {} +- {}'.format(curr_avg_reward, np.std(episodic_rewards)))
    env.close()


def test_agent(agent_name, n_steps=100):
    prev_episode_reward = -np.inf
    with T.no_grad():
        episodic_rewards = []
        for _ in range(n_steps):
            real_obs = []
            vae_obs = []
            observation = env.reset()
            state = convert_observation(observation)
            cuda_state = state.to(device)
            z = VAE_network.forward(cuda_state, encoder=True, decoder=False)
            done = False
            episode_reward = 0
            while not done:
                real_obs.append(observation)
                vae_obs.append(VAE_network.decode(z).detach().cpu().numpy())
                action = agent.choose_action(z, reparameterize=False)
                new_observation, reward, done, info = env.step(action)
                new_state = convert_observation(new_observation)
                cuda_new_state = new_state.to(device)
                new_z = VAE_network.forward(cuda_new_state, encoder=True, decoder=False)
                z = new_z
                observation = new_observation
                episode_reward += reward
                if episode_reward < -500:
                    print('Policy needs to train')
                    break
                env.render()

            print('Episode: {}, Reward: {}'.format(_, episode_reward))
            episodic_rewards.append(episode_reward)
            if prev_episode_reward < episode_reward:
                final_real_obs = real_obs
                final_vae_obs = vae_obs
                prev_episode_reward = episode_reward

        print('Evaluation Done')
        print('Mean-Return: {}, Std-Return: {}'.format(np.mean(episodic_rewards), np.std(episodic_rewards)))
        test_arr = np.array(final_real_obs)
        pred_arr = np.array(final_vae_obs)
        get_video(test_arr, pred_arr.squeeze(axis=1), agent_name)
        plt.plot(range(len(episodic_rewards)), episodic_rewards, label='Evaluation-Reward')
        plt.xlabel('Evaluation episodes')
        plt.ylabel('Reward')
        plt.show()
        env.close()


if __name__ == "__main__":
    opts = [opt for opt in sys.argv[1:] if opt.startswith("--")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    for i, opt in enumerate(opts):
        if opt == '--model_dir':
            model_path = os.getcwd() + args[i]
        if opt == '--vae_model_name':
            vae_name = args[i]
        if opt == '--sac_model_name':
            sac_name = args[i]
        if opt == '--agent_pipeline':
            flag = args[i]

    if not os.path.exists(model_path):
        print('Making directory: ', model_path)
        os.mkdir(model_path)
    else:
        print('Model directory exists.')

    env = gym.make("CarRacing-v0")
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    try:
        print(T.cuda.get_device_name(device))
    except Exception as e:
        print(e)

    vae_model_path = model_path + '/vae/' + vae_name
    if os.path.exists(vae_model_path):
        print('VAE file found.')
        VAE_network = VAE().to(device)
        VAE_network.load_state_dict(T.load(vae_model_path + '/' + vae_name + '.pth', map_location=device))
        VAE_network.eval()
        for param in VAE_network.parameters():
            param.requires_grad = False

        sac_model_path = model_path + '/sac/' + sac_name
        load_model_flag = False
        writer = SummaryWriter('logs/sac/' + sac_name + '_runs')
        agent = SACAgent(device=device, model_dir=sac_model_path)
        for param in agent.target_critic.parameters():
            param.requires_grad = False

        if os.path.exists(sac_model_path):
            print('SAC model file found.')
            files = [file for file in os.listdir(sac_model_path)]
            if len(files) > 0:
                print('Uploading state dict.')
                agent.load_model()
                load_model_flag = True
        else:
            print('SAC model file not found. Creating directory')
            os.mkdir(sac_model_path)

        T.manual_seed(1234)
        np.random.seed(1234)
        env.seed(1234)

        if flag == 'train_agent':
            train_agent(load_model_flag)
        elif flag == 'test_agent':
            test_agent(sac_name, n_steps=5)
        else:
            print('Invalid flag parameter.')
    else:
        print('VAE model file not found. Exiting program.')