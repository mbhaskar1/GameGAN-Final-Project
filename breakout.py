import gym
import cv2
import numpy as np
import random
from dqn_network import DQN
import torch.nn as nn
import torch.optim as optim
import torch

NUM_ACTIONS = 3  # Pong actions are {1: nothing, 2: up, 3: down}
IMAGE_SIZE = 84
EPOCHS = 1000
EPISODES_PER_EPOCH = 2
STACK_SIZE = 4
REPLAY_MEMORY_CAPACITY = 10000
REPLAY_MEMORY_MINIMUM = 1000
BATCH_SIZE = 32
INITIAL_EXPLORE_PROBABILITY = 1
MINIMUM_EXPLORE_PROBABILITY = 0.1
DELTA_EXPLORE_PROBABILITY = 0.001
DISCOUNT_FACTOR = 0.99
GPU = True


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
    return cv2.cvtColor(cv2.resize(image[34:-16, :, :], (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_RGB2GRAY)


replay_memory = ReplayMemory()
dqn = DQN().double()
target_dqn = DQN().double()
target_dqn.load_state_dict(dqn.state_dict())

device = torch.device('cuda:0')
if GPU:
    dqn.to(device)
    target_dqn.to(device)

explore_probability = INITIAL_EXPLORE_PROBABILITY

env = gym.make('Pong-v0')

loss_function = nn.MSELoss()
optimizer = optim.Adam(params=dqn.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    target_dqn.load_state_dict(dqn.state_dict())
    for episode in range(EPISODES_PER_EPOCH):
        observation = env.reset()
        history = np.zeros((STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE))
        history[0, :, :] = preprocess(observation)
        total_reward = 0
        step = 0
        while True:
            step += 1
            if step % 1 == 0:
                print(f'Step {step}')
            # With probability ε select a random action
            if random.random() < explore_probability:
                action = random.randint(1, 3)
            # Otherwise select using DQN
            else:
                t = torch.tensor(history).unsqueeze(0)
                if GPU:
                    t = t.to(device)
                _, indices = dqn.forward(t).max(1)
                action = indices[0] + 1
            # Execute action in emulator and observe reward and image
            observation, reward, done, info = env.step(env.action_space.sample())
            total_reward += reward
            # Update history
            previous_history = np.copy(history)
            history[1:, :, :] = history[:STACK_SIZE - 1, :, :]
            history[0, :, :] = preprocess(observation)
            # Store transition in replay memory
            if action > 3 or action < 1:
                print('Something is not right')
            replay_memory.add_memory((previous_history, action, reward, np.copy(history), done))
            if len(replay_memory) >= REPLAY_MEMORY_MINIMUM:
                # Sample random mini-batch of transitions
                sample = replay_memory.sample_batch()
                # Calculate target Q-values
                next_histories = torch.tensor(np.stack([s[3] for s in sample]), dtype=torch.double)
                histories = torch.tensor(np.stack([s[0] for s in sample]), dtype=torch.double)
                y = torch.tensor([s[2] for s in sample], dtype=torch.double)
                if GPU:
                    next_histories = next_histories.to(device)
                    histories = histories.to(device)
                    y = y.to(device)
                max_future_Qs, _ = torch.max(target_dqn.forward(next_histories), 1)
                if GPU:
                    max_future_Qs = max_future_Qs.to(device)
                for i in range(BATCH_SIZE):
                    if not sample[i][4]:
                        y[i] += DISCOUNT_FACTOR * max_future_Qs[i]
                # Calculate loss and perform gradient descent
                optimizer.zero_grad()
                output = dqn.forward(histories)
                predicted_Qs = torch.empty(BATCH_SIZE, dtype=torch.double)
                if GPU:
                    predicted_Qs = predicted_Qs.to(device)
                for i in range(BATCH_SIZE):
                    predicted_Qs[i] = output[i, sample[i][1] - 1]
                loss = loss_function(y, predicted_Qs)
                loss.backward()
                optimizer.step()
            if done:
                print(f'Total Reward: {total_reward}')
                break

        explore_probability = max(MINIMUM_EXPLORE_PROBABILITY, explore_probability - DELTA_EXPLORE_PROBABILITY)
        print(f'Explore Probability: {explore_probability}')

    torch.save(dqn, f'model-{epoch}')


# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(10000):
#         env.render()
#         # print(observation)
#         print(env.action_space)
#         action = 1
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
env.close()