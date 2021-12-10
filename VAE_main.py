import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as t
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import os
import gym
import sys
from tqdm import tqdm
from gym.envs.box2d import car_racing
from gym.envs.box2d.car_dynamics import Car
import numpy as np
from collections import OrderedDict
from PIL import Image as PIL_Image
import cv2
from ipywidgets import interact, interact_manual
from IPython.display import clear_output, display
from IPython.display import Image
import matplotlib.pyplot as plt
import multiprocessing as mp

from VAE import VAE


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


def simulate_batch(batch_num, batch_size=16, time_steps=150, render=True):
    env = car_racing.CarRacing()
    obs_data = []
    action = env.action_space.sample()
    cur_dir = os.getcwd() + '/dataset'
    for i_episode in range(batch_size):
        observation = env.reset()
        position = np.random.randint(len(env.track))
        env.car = Car(env.world, *env.track[position][1:4])
        for i_step in range(time_steps):
            if render:
                env.render()
            action = generate_action(action)
            observation, reward, done, info = env.step(action)
            obs_data.append(observation)

    print("Saving dataset for batch {}".format(batch_num))
    np.save(cur_dir + '/obs_data_AE_{}'.format(batch_num), obs_data)
    env.close()


def gather_data(arr, data):
    return data if arr.shape[0] == 0 else np.concatenate((arr, data), axis=0)


def get_data_array():
    print('Fetching data arrays')
    dataset_dir = os.getcwd() + '/dataset'
    train_array = np.array([])
    test_array = np.array([])
    for np_file in os.listdir(dataset_dir):
        file_num = int(np_file.split('_')[-1][0])
        data_array = np.load(dataset_dir + '/' + np_file)
        if file_num < 7:
            train_array = gather_data(train_array, data_array)
        elif 7 <= file_num:
            test_array = gather_data(test_array, data_array)

    print('Train-data shape: {} \nTest-data shape: {}'.format(train_array.shape, test_array.shape))
    return train_array, test_array


def generate_dataloaders(data, batch_size=64):
    print('Generating dataloaders')
    np.random.seed(1111)
    data_indices = list(range(data.shape[0]))
    np.random.shuffle(data_indices)
    x = T.einsum("nhwc -> nchw", T.tensor(data)).type('torch.FloatTensor')
    dataset = TensorDataset(x, x)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def get_video(test, pred):
    size = (192, 96)
    video_fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    video = cv2.VideoWriter(vae_name + '.mp4', video_fourcc, 60, size)
    print('Converting predictions to video')
    for i in tqdm(range(test.shape[0])):
        arr1 = np.einsum("chw -> hwc", test[i, :, :, :])
        arr2 = np.einsum("chw -> hwc", pred[i, :, :, :])
        arr1 = np.asarray(arr1, dtype=np.uint8)
        arr2 = np.asarray(arr2, dtype=np.uint8)
        arr = cv2.hconcat([arr1, arr2])
        video.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    video.release()


def train_vae(epochs=200):
    print('Training begins')
    mean_train_loss = []
    mean_test_loss = []
    curr_loss = 0
    prev_loss = np.inf
    for epoch in range(epochs):
        train_loss = []
        test_loss = []
        for i, train_data in enumerate(train_dataloader):
            x_train, y_train = train_data
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            y_train_pred, mean, logvar = VAE_network.forward(x_train)
            batch_size = x_train.shape[0]
            reconstruction_loss = F.mse_loss(y_train_pred, y_train, reduction='sum') / batch_size
            KLD_loss = -0.5 * T.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch_size
            loss = reconstruction_loss + 3 * KLD_loss
            train_loss.append(loss)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Train loss', loss.item(), epoch * len(train_dataloader) + i)

        train_mean_loss = np.mean([i.item() for i in train_loss])
        mean_train_loss.append(train_mean_loss)
        with T.no_grad():
            for i, test_data in enumerate(test_dataloader):
                x_test, y_test = test_data
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                y_pred, mean, logvar = VAE_network.forward(x_test)
                loss = F.mse_loss(y_pred, y_test, reduction='sum') / batch_size
                test_loss.append(loss)
                writer.add_scalar('Test loss', loss.item(), epoch * len(test_dataloader) + i)

            test_mean_loss = np.mean([i.item() for i in test_loss])
            mean_test_loss.append(test_mean_loss)
            print('Epoch, Train-loss, Valid-loss: {}, {}, {}'.format(epoch + 1,
                                                                     train_mean_loss,
                                                                     test_mean_loss))
            scheduler.step(test_mean_loss)
            curr_loss = test_mean_loss
            if curr_loss < prev_loss and epoch > 50 and epoch % 10 == 0:
                print('Saving Model at epoch {}'.format(epoch + 1))
                model_path = os.getcwd() + '/tmp'
                T.save(VAE_network.state_dict(), vae_model_path)
                prev_loss = curr_loss


def test_vae():
    print('Running VAE on test-set')
    test_list = []
    pred_list = []
    for i, test_data in enumerate(test_dataloader):
        x_test, y_test = test_data
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        y_pred, mean, logvar = VAE_network.forward(x_test)
        test_point = x_test.detach().cpu().numpy()
        pred_point = y_pred.detach().cpu().numpy()
        test_list.append(test_point[0])
        pred_list.append(pred_point[0])

    test_arr = np.array(test_list)
    pred_arr = np.array(pred_list)
    get_video(test_arr, pred_arr)


if __name__ == "__main__":
    opts = [opt for opt in sys.argv[1:] if opt.startswith("--")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    data_flag = 'n'
    for i, opt in enumerate(opts):
        if opt == '--model_dir':
            model_path = os.getcwd() + args[i]
        if opt == '--vae_model_name':
            vae_name = args[i]
        if opt == '--collect_data':
            data_flag = args[i]
        if opt == '--vae_pipeline':
            flag = args[i]

    if not os.path.exists(model_path):
        print('Making directory: ', model_path)
        os.mkdir(model_path)
    else:
        print('Model directory exists.')

    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    try:
        print(T.cuda.get_device_name(device))
    except Exception as e:
        print(e)

    if data_flag == 'y':
        print('Collecting data-points for VAE')
        batch_num = 5
        with mp.Pool(mp.cpu_count()) as p:
            for _ in tqdm(p.imap_unordered(simulate_batch, list(range(batch_num))), total=batch_num):
                pass
    elif data_flag == 'n':
        print('Using data from dataset directory')

    train_array, test_array = get_data_array()
    train_dataloader = generate_dataloaders(train_array)
    test_dataloader = generate_dataloaders(test_array, batch_size=1)
    vae_model_path = model_path + '/vae/' + vae_name
    VAE_network = VAE().to(device)
    writer = SummaryWriter('logs/vae/' + vae_name + '_runs')
    lr = 1e-4
    lambda_reg = 2
    optimizer = optim.Adam(VAE_network.parameters(), weight_decay=lambda_reg, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    if os.path.exists(vae_model_path):
        print('VAE file found')
        VAE_network.load_state_dict(T.load(vae_model_path + '/' + vae_name + '.pth', map_location=device))
        VAE_network.eval()
    else:
        os.mkdir(vae_model_path)
        print('VAE model file not found. Training from scratch.')

    if flag == 'train_vae':
        train_vae()
    elif flag == 'test_vae':
        test_vae()




