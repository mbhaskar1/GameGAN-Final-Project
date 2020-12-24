import gym
import cv2
import numpy as np
import random
from dqn_network import DQN
import torch.nn as nn
import torch.optim as optim
import torch
from atari_wrappers import *
import time

NUM_ACTIONS = 3  # Pong actions are {1: nothing, 2: up, 3: down}
IMAGE_SIZE = 84
EPOCHS = 20
EPISODES_PER_EPOCH = 50
STACK_SIZE = 4
REPLAY_MEMORY_CAPACITY = 10000
REPLAY_MEMORY_MINIMUM = 10000
TRAIN_EVERY = 1
BATCH_SIZE = 32 * TRAIN_EVERY
EXPLORE_PROBABILITIES = [0]
EXPLORE_PROBABILITY_DECAY_EPISODES = 1000000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.0001
TARGET_REPLACE_ITER = 1000
GPU = True
RENDER = False


def preprocess(image):
    return (1/255.0) * cv2.cvtColor(cv2.resize(image[34:-16, :, :], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)


dqn = torch.load('dqn_best')

device = torch.device('cuda:0')
if GPU:
    dqn.to(device)

env = make_atari('PongNoFrameskip-v4')
env = FireResetEnv(env)

loss_function = nn.MSELoss()
optimizer = optim.Adam(params=dqn.parameters(), lr=LEARNING_RATE)

counter = 0
step_counter = 0
for epoch in range(EPOCHS):
    dataset = []
    for episode in range(EPISODES_PER_EPOCH):
        observation = env.reset()
        history = np.zeros((STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE))
        history[0, :, :] = preprocess(observation)
        total_reward = 0
        while True:
            step_counter += 1

            if random.random() < EXPLORE_PROBABILITIES[episode % len(EXPLORE_PROBABILITIES)]:
                action = random.randint(1, 3)
            # Otherwise select using DQN
            else:
                t = torch.tensor(history).unsqueeze(0)
                if GPU:
                    t = t.to(device)
                indices = dqn.forward(t).max(1)[1].cpu().data.numpy()
                action = indices[0] + 1
            # Execute action in emulator and observe reward and image
            observation, reward, done, info = env.step(action)
            reward = min(max(reward, -1), 1)  # clip reward to [-1, 1]
            total_reward += reward
            # Update history
            history[1:, :, :] = history[:STACK_SIZE - 1, :, :]
            history[0, :, :] = preprocess(observation)

            env.render()
            time.sleep(0.03)
            if done:
                print(f'Episode: {episode + 1}')
                print(f'Total Reward: {total_reward}')
                break

env.close()