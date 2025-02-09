import gym
import cv2
import numpy as np
import random
from dqn_network import DQN
import torch.nn as nn
import torch.optim as optim
import torch
import os
from atari_wrappers import *

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

NUM_ACTIONS = 3  # Pong actions are {1: nothing, 2: up, 3: down}
IMAGE_SIZE = 84
EPOCHS = 1000
EPISODES_PER_EPOCH = 50
STACK_SIZE = 4
REPLAY_MEMORY_CAPACITY = 10000
REPLAY_MEMORY_MINIMUM = 10000
TRAIN_EVERY = 1
BATCH_SIZE = 32 * TRAIN_EVERY
INITIAL_EXPLORE_PROBABILITY = 1
MINIMUM_EXPLORE_PROBABILITY = 0.02
EXPLORE_PROBABILITY_DECAY_EPISODES = 1000000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.0001
TARGET_REPLACE_ITER = 1000
GPU = True
RENDER = False


class ReplayMemory:
    def __init__(self):
        self.memory_dataset = []
        self.i = 0

    def add_memory(self, memory):
        if self.i >= len(self.memory_dataset):
            self.memory_dataset.append(memory)
        else:
            self.memory_dataset[self.i] = memory
        self.i = (self.i + 1) % REPLAY_MEMORY_CAPACITY

    def sample_batch(self):
        sample = random.sample(self.memory_dataset, BATCH_SIZE)
        return sample

    def __len__(self):
        return len(self.memory_dataset)


def preprocess(image):
    return (1/255.0) * cv2.cvtColor(cv2.resize(image[34:-16, :, :], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)


replay_memory = ReplayMemory()
dqn = DQN().double()
target_dqn = DQN().double()
target_dqn.load_state_dict(dqn.state_dict())

device = torch.device('cuda:0')
if GPU:
    dqn.to(device)
    target_dqn.to(device)

explore_probability = INITIAL_EXPLORE_PROBABILITY

env = make_atari('PongNoFrameskip-v4')
env = EpisodicLifeEnv(env)
env = FireResetEnv(env)

loss_function = nn.MSELoss()
optimizer = optim.Adam(params=dqn.parameters(), lr=LEARNING_RATE)

counter = 0
episode_counter = 0
step_counter = 0
for epoch in range(EPOCHS):
    for episode in range(EPISODES_PER_EPOCH):
        episode_counter += 1
        observation = env.reset()
        history = np.zeros((STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE))
        history[0, :, :] = preprocess(observation)
        total_reward = 0
        while True:
            explore_probability = INITIAL_EXPLORE_PROBABILITY - (INITIAL_EXPLORE_PROBABILITY - MINIMUM_EXPLORE_PROBABILITY) * min(1, step_counter / EXPLORE_PROBABILITY_DECAY_EPISODES)
            step_counter += 1
            # With probability ε select a random action
            if random.random() < explore_probability:
                action = random.randint(2, 3)
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
            previous_history = np.copy(history)
            history[1:, :, :] = history[:STACK_SIZE - 1, :, :]
            history[0, :, :] = preprocess(observation)
            # Store transition in replay memory
            if action > 3 or action < 1:
                print('Something is not right')
            replay_memory.add_memory((previous_history, action, reward, np.copy(history), done))
            if RENDER:
                env.render()
            if step_counter % TRAIN_EVERY == 0 and len(replay_memory) >= REPLAY_MEMORY_MINIMUM:
                if counter % (TARGET_REPLACE_ITER // TRAIN_EVERY) == 0:
                    target_dqn.load_state_dict(dqn.state_dict())
                counter += 1
                # Sample random mini-batch of transitions
                sample = replay_memory.sample_batch()
                # Calculate target Q-values
                next_histories = torch.tensor(np.stack([s[3] for s in sample]), dtype=torch.double)
                histories = torch.tensor(np.stack([s[0] for s in sample]), dtype=torch.double)
                y = torch.tensor([s[2] for s in sample], dtype=torch.double).view(BATCH_SIZE, 1)
                if GPU:
                    next_histories = next_histories.to(device)
                    histories = histories.to(device)
                    y = y.to(device)
                future_Qs = target_dqn.forward(next_histories).detach()
                max_future_Qs = torch.max(future_Qs, 1)[0].view(BATCH_SIZE, 1)
                if GPU:
                    max_future_Qs = max_future_Qs.to(device)
                # Using max future Q regardless of whether game is done (game ending should be similar to states during game)
                y += DISCOUNT_FACTOR * max_future_Qs
                # Calculate loss and perform gradient descent
                actions = torch.tensor([s[1] - 1 for s in sample], dtype=torch.long).view(BATCH_SIZE, 1)
                if GPU:
                    actions = actions.to(device)
                predicted_Qs = dqn.forward(histories).gather(1, actions)
                if GPU:
                    predicted_Qs = predicted_Qs.to(device)
                loss = loss_function(y, predicted_Qs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if done:
                    print(f'Episode: {episode_counter}')
                    print(f'Step: {step_counter}')
                    print(f'Total Reward: {total_reward}')
                    print(f'Explore Probability: {explore_probability}')
                    break
            if done:
                print(f'Episode: {episode_counter}')
                print(f'Step: {step_counter}')
                print(f'Total Reward: {total_reward}')
                print(f'Explore Probability: {explore_probability}')
                break

    torch.save(dqn, f'model-{epoch}')

env.close()