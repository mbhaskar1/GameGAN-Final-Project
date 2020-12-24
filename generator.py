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

    def get_batch(self, batch_size=32):
        images = self.image_dataset[self.current_index:min(self.current_index + batch_size, self.image_dataset.shape[0] - 1)]
        actions = self.action_dataset[self.current_index:min(self.current_index + batch_size, self.action_dataset.shape[0] - 1)]
        self.current_index += batch_size
        data_exhausted = self.current_index >= self.image_dataset.shape[0]
        return images, actions, data_exhausted


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv2d1 = nn.Conv2d(NUM_INPUT_IMAGES, 32, 3, 2, 1)  # 3x84x84 -> 8x41x41
        self.conv2d2 = nn.Conv2d(32, 32, 3, 1, 0)  # 8x41x41 -> 8x41x41
        self.conv2d3 = nn.Conv2d(32, 32, 3, 2, 0)  # 8x41x41 -> 16x20x20
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


device = torch.device('cuda:0')

generator = Generator().double()
if GPU:
    generator = generator.to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(params=generator.parameters(), lr=0.001)

for repeat in range(10):
    for epoch in range(NUM_EPOCHS):
        dataset = Dataset(f'Data/Data-{epoch}.npy')

        avg_loss = 0
        index = 0
        exhausted_data = False
        while not exhausted_data:
            index += 1

            # Sample a batch and train on it
            images, actions, exhausted_data = dataset.get_batch()
            input_images = torch.tensor(images[:, :-1, :].reshape((images.shape[0], NUM_INPUT_IMAGES, 84, 84)))
            target_images = torch.tensor(images[:, -1, :].reshape((images.shape[0], 1, 84, 84)))
            actions = torch.tensor(actions, dtype=torch.double)
            if GPU:
                input_images = input_images.to(device)
                target_images = target_images.to(device)
                actions = actions.to(device)
            optimizer.zero_grad()
            Z = torch.randn((images.shape[0], NZ), dtype=torch.double)
            if GPU:
                Z = Z.to(device)
            outputs = generator.forward(input_images, actions, Z)
            loss = loss_function(outputs, target_images)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if index % 10 == 0:
                print(avg_loss / index)

        if avg_loss / index <= 0.00012:
            for g in optimizer.param_groups:
                g['lr'] = 0.0001
                print('updated learning rate')

        images = [input_images[:, i, :, :] for i in range(NUM_INPUT_IMAGES)]
        output = outputs[:, 0, :, :]
        target = target_images[:, 0, :, :]
        if epoch % 5 == 0:
            for im in range(min(1, images[0].shape[0])):
                for i in range(NUM_INPUT_IMAGES):
                    cv2.imwrite(f'Sample-Input-{epoch}-{i}.png', np.uint8(255.0 * images[i][im, :, :].cpu().detach().numpy()))
                cv2.imwrite(f'Sample-Output-{epoch}.png', np.uint8(255.0 * output[im, :, :].cpu().detach().numpy()))
                cv2.imwrite(f'Sample-Target-{epoch}.png', np.uint8(255.0 * target[im, :, :].cpu().detach().numpy()))
            torch.save(generator, f'generator-model-{epoch}.mdl')

        print(f'Epoch: {epoch + repeat * 60} Average Loss: {avg_loss / index}')



