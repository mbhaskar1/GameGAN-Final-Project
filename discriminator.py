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
        # print(x.shape)
        x = F.leaky_relu(self.conv2d2(x), negative_slope=0.2)
        # print(x.shape)
        x = F.leaky_relu(self.conv2d3(x), negative_slope=0.2)
        # print(x.shape)
        x = F.leaky_relu(self.conv2d4(x), negative_slope=0.2)
        # print(x.shape)
        x = F.leaky_relu(self.conv2d5(x), negative_slope=0.2)
        # print(x.shape)
        x = x.view(-1, 32*8*8)
        # print(x.shape)
        x = F.leaky_relu(self.linear1(x), negative_slope=0.2)
        # print(x.shape)
        x = torch.cat([x, action, latent_vector], dim=1)
        # print(x.shape)
        x = F.leaky_relu(self.linear2(x), negative_slope=0.2)
        # print(x.shape)
        x = x.view(-1, 256, 7, 7)
        # print(x.shape)
        x = F.leaky_relu(self.convT2d1(x), negative_slope=0.2)
        # print(x.shape)
        x = F.leaky_relu(self.convT2d2(x), negative_slope=0.2)
        # print(x.shape)
        x = F.leaky_relu(self.convT2d3(x), negative_slope=0.2)
        # print(x.shape)
        x = F.leaky_relu(self.convT2d4(x), negative_slope=0.2)
        # print(x.shape)
        x = F.leaky_relu(self.convT2d5(x), negative_slope=0.2)
        # print(x.shape)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv2d1 = nn.Conv2d(1, 16, 5, 2, 0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2d2 = nn.Conv2d(16, 32, 5, 2, 0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2d3 = nn.Conv2d(32, 32, 3, 2, 0)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv2d4 = nn.Conv2d(32, 32, 3, 2, 0)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv2dO1 = nn.Conv2d(1, 16, 5, 2, 0)
        self.bnO1 = nn.BatchNorm2d(16)
        self.conv2dO2 = nn.Conv2d(16, 32, 5, 2, 0)
        self.bnO2 = nn.BatchNorm2d(32)
        self.conv2dO3 = nn.Conv2d(32, 32, 3, 2, 0)
        self.bnO3 = nn.BatchNorm2d(32)
        self.conv2dO4 = nn.Conv2d(32, 32, 3, 2, 0)
        self.bnO4 = nn.BatchNorm2d(32)

        self.action_encoder = nn.Linear(3, 16)

        self.conv2d_combine = nn.Conv2d(32 * (NUM_INPUT_IMAGES + 1), 32, 3, 1, 0)
        self.bn_combine = nn.BatchNorm2d(32)

        self.linear1 = nn.Linear(32, 32)
        self.bn_linear = nn.BatchNorm1d(32)
        self.linear2 = nn.Linear(32, 1)

    def forward(self, images, action, new_image):
        batch_size = images.shape[0]
        images = images.view((batch_size * NUM_INPUT_IMAGES, 1, 84, 84))
        images = F.leaky_relu(self.bn1(self.conv2d1(images)), 0.2)
        images = F.leaky_relu(self.bn2(self.conv2d2(images)), 0.2)
        images = F.leaky_relu(self.bn3(self.conv2d3(images)), 0.2)
        images = F.leaky_relu(self.bn4(self.conv2d4(images)), 0.2)
        images = images.view((batch_size, 32 * NUM_INPUT_IMAGES, images.shape[2], images.shape[2]))
        # print(images.shape)

        new_image = F.leaky_relu(self.bnO1(self.conv2dO1(new_image)), 0.2)
        new_image = F.leaky_relu(self.bnO2(self.conv2dO2(new_image)), 0.2)
        new_image = F.leaky_relu(self.bnO3(self.conv2dO3(new_image)), 0.2)
        new_image = F.leaky_relu(self.bnO4(self.conv2dO4(new_image)), 0.2)
        # print(new_image.shape)

        combined = torch.cat([images, new_image], 1)
        # print(combined.shape)
        combined = F.leaky_relu(self.bn_combine(self.conv2d_combine(combined)), 0.2)
        # print(combined.shape)
        combined = combined.view((batch_size, 32))

        output = F.leaky_relu(self.bn_linear(self.linear1(combined)), 0.2)
        output = self.linear2(output)
        # print(output.shape)
        return output


device = torch.device('cuda:0')

generator = torch.load('pretrained_G.mdl')
discriminator = Discriminator().double()
if GPU:
    generator = generator.to(device)
    discriminator = discriminator.to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=discriminator.parameters(), lr=0.0001)

for repeat in range(10):
    for epoch in range(NUM_EPOCHS):
        dataset = Dataset(f'Data/Data-{epoch}.npy')

        avg_loss = 0
        index = 0
        exhausted_data = False
        while not exhausted_data:
            index += 1

            # Sample a batch and train on it
            images, actions, exhausted_data = dataset.get_batch(batch_size=BATCH_SIZE)
            input_images = torch.tensor(images[:, :-1, :].reshape((images.shape[0], NUM_INPUT_IMAGES, 84, 84)))
            output_images = torch.tensor(images[:, -1, :].reshape((images.shape[0], 1, 84, 84)))
            actions = torch.tensor(actions, dtype=torch.double)
            if GPU:
                input_images = input_images.to(device)
                output_images = output_images.to(device)
                actions = actions.to(device)
            optimizer.zero_grad()
            Z = torch.randn((images.shape[0], NZ), dtype=torch.double)
            if GPU:
                Z = Z.to(device)
            output_images[:BATCH_SIZE // 2] = generator.forward(input_images[:BATCH_SIZE//2], actions[:BATCH_SIZE//2], Z[:BATCH_SIZE//2])
            predictions = discriminator.forward(input_images, actions, output_images)
            labels = torch.ones((BATCH_SIZE, 1))
            labels[:BATCH_SIZE//2, :] = 0
            if GPU:
                labels = labels.to(device)

            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if index % 10 == 0:
                print(avg_loss / index)

        if epoch % 5 == 0:
            torch.save(discriminator, f'discriminator-model-{repeat * NUM_EPOCHS + epoch}.mdl')

        print(f'Epoch: {epoch + repeat * NUM_EPOCHS} Average Loss: {avg_loss / index}')



