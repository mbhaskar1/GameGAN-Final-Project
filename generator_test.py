#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from keras.utils.np_utils import to_categorical
import numpy as np
import cv2

NUM_INPUT_IMAGES = 3
NZ = 8
NUM_EPOCHS = 60
GPU = True
BATCH_SIZE = 32
MAE_LOSS_MULTIPLIER = 10


class Dataset:
    def __init__(self, filepath, shuffle=True):
        self.raw_data = np.load(f'{filepath}')
        self.image_dataset = np.empty((self.raw_data.shape[0] - NUM_INPUT_IMAGES, NUM_INPUT_IMAGES + 1, 84 * 84))
        for i in range(NUM_INPUT_IMAGES):
            self.image_dataset[:, i, :] = self.raw_data[i:-NUM_INPUT_IMAGES + i, :-1]
        self.image_dataset[:, NUM_INPUT_IMAGES, :] = self.raw_data[NUM_INPUT_IMAGES:, :-1]
        self.action_dataset = to_categorical(self.raw_data[NUM_INPUT_IMAGES:, -1] - 1)
        if shuffle:
            np.random.shuffle(self.image_dataset)
        self.current_index = 0

    def get_batch(self, batch_size=32, minimum_batch_size=32):
        images = self.image_dataset[self.current_index:min(self.current_index + batch_size, self.image_dataset.shape[0])]
        actions = self.action_dataset[self.current_index:min(self.current_index + batch_size, self.action_dataset.shape[0])]
        self.current_index += batch_size
        data_exhausted = self.current_index >= self.image_dataset.shape[0] - minimum_batch_size + 1
        return images, actions, data_exhausted


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv2d1 = nn.Conv2d(NUM_INPUT_IMAGES, 32, 3, 2, 1)
        self.conv2d2 = nn.Conv2d(32, 32, 3, 1, 0)
        self.conv2d3 = nn.Conv2d(32, 32, 3, 2, 0)
        self.conv2d4 = nn.Conv2d(32, 32, 3, 1, 0)
        self.conv2d5 = nn.Conv2d(32, 32, 3, 2, 0)
        self.linear1 = nn.Linear(32*8*8, 256)
        self.linear2 = nn.Linear(256 + 3 + NZ, 256*7*7)
        self.convT2d1 = nn.ConvTranspose2d(256, 256, 3, 1, 0, 0)
        self.convT2d2 = nn.ConvTranspose2d(256, 128, 3, 2, 0, 0)
        self.convT2d3 = nn.ConvTranspose2d(128, 64, 4, 2, 0, 0)
        self.convT2d4 = nn.ConvTranspose2d(64, 32, 4, 2, 0, 0)
        self.convT2d5 = nn.ConvTranspose2d(32, 1, 3, 1, 0, 0)

    def forward(self, image, action, latent_vector):
        x = F.leaky_relu(self.conv2d1(image), negative_slope=0.2)
        x = F.leaky_relu(self.conv2d2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2d3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2d4(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2d5(x), negative_slope=0.2)
        x = x.view(-1, 32*8*8)
        x = F.leaky_relu(self.linear1(x), negative_slope=0.2)
        x = torch.cat([x, action, latent_vector], dim=1)
        x = F.leaky_relu(self.linear2(x), negative_slope=0.2)
        x = x.view(-1, 256, 7, 7)
        x = F.leaky_relu(self.convT2d1(x), negative_slope=0.2)
        x = F.leaky_relu(self.convT2d2(x), negative_slope=0.2)
        x = F.leaky_relu(self.convT2d3(x), negative_slope=0.2)
        x = F.leaky_relu(self.convT2d4(x), negative_slope=0.2)
        x = F.leaky_relu(self.convT2d5(x), negative_slope=0.2)
        return x


generator = torch.load('pretrained_G.mdl')
generator = generator.cpu()
dataset = Dataset(f'Testing/Data-1.npy')
# Sample a batch and train on it
images, actions, exhausted_data = dataset.get_batch(batch_size=BATCH_SIZE, minimum_batch_size=32)
input_images = torch.tensor(images[:, :-1, :].reshape((images.shape[0], NUM_INPUT_IMAGES, 84, 84)))
actions = torch.tensor(actions, dtype=torch.double)

Z = torch.randn((images.shape[0], NZ), dtype=torch.double)

real_images = torch.tensor(images[:, -1, :].reshape((images.shape[0], 1, 84, 84)))
generated_images = generator.forward(input_images, actions, Z)

sample_inputs = [np.uint8(255.0 * input_images[:, i, :, :].detach().numpy()) for i in range(NUM_INPUT_IMAGES)]
sample_outputs = np.uint8(255.0 * generated_images[:, 0, :, :].detach().numpy())
target_outputs = np.uint8(255.0 * real_images[:, 0, :, :].detach().numpy())
for im in range(sample_inputs[0].shape[0]):
    image = np.concatenate([*[np.pad(s[im], 1, mode='constant') for s in sample_inputs],
                            np.pad(sample_outputs[im], 1, mode='constant'),
                            np.pad(target_outputs[im], 1, mode='constant')], axis=1)
    cv2.imshow('Sample', image)
    cv2.waitKey()
