import gym
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

EPOCHS = 1000
EPISODES_PER_EPOCH = 50
REPLAY_MEMORY_CAPACITY = 2000
REPLAY_MEMORY_MINIMUM = 2000
BATCH_SIZE = 32
INITIAL_EXPLORE_PROBABILITY = 0.1
MINIMUM_EXPLORE_PROBABILITY = 0.1
DELTA_EXPLORE_PROBABILITY = 0.001
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
TARGET_REPLACE_ITER = 100
GPU = False


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


class CartpoleDQN(nn.Module):
    def __init__(self):
        super(CartpoleDQN, self).__init__()
        self.linear1 = nn.Linear(4, 50)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2 = nn.Linear(50, 2)
        self.linear2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


replay_memory = ReplayMemory()
dqn = CartpoleDQN().double()
target_dqn = CartpoleDQN().double()
target_dqn.load_state_dict(dqn.state_dict())

device = torch.device('cuda:0')
if GPU:
    dqn.to(device)
    target_dqn.to(device)

explore_probability = INITIAL_EXPLORE_PROBABILITY

env = gym.make('CartPole-v0')
env = env.unwrapped

loss_function = nn.MSELoss()
optimizer = optim.Adam(params=dqn.parameters(), lr=LEARNING_RATE)

counter = 0
print('Generating Data')
episode_counter = 0
step_counter = 0
for epoch in range(EPOCHS):
    for episode in range(EPISODES_PER_EPOCH):
        episode_counter += 1
        observation = env.reset()
        observation = np.reshape(observation, (1, 4))
        total_reward = 0
        while True:
            step_counter += 1
            # With probability ε select a random action
            if random.random() < explore_probability:
                action = random.randint(0, 1)
            # Otherwise select using DQN
            else:
                t = torch.tensor(observation)
                if GPU:
                    t = t.to(device)
                indices = dqn.forward(t).max(1)[1].data.numpy()
                action = indices[0]
            # Execute action in emulator and observe reward and image
            new_observation, reward, done, info = env.step(action)
            x, x_dot, theta, theta_dot = new_observation
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            new_observation = np.reshape(new_observation, (1, 4))

            total_reward += reward
            # Store transition in replay memory
            replay_memory.add_memory((np.copy(observation), action, reward, np.copy(new_observation), done))
            if len(replay_memory) >= REPLAY_MEMORY_MINIMUM:
                env.render()
                if counter % TARGET_REPLACE_ITER == 0:
                    target_dqn.load_state_dict(dqn.state_dict())
                    print('updating target')
                counter += 1
                # Sample random mini-batch of transitions
                sample = replay_memory.sample_batch()
                # Calculate target Q-values
                next_histories = torch.tensor(np.concatenate([s[3] for s in sample]), dtype=torch.double)
                histories = torch.tensor(np.concatenate([s[0] for s in sample]), dtype=torch.double)
                y = torch.tensor([s[2] for s in sample], dtype=torch.double).view(BATCH_SIZE, 1)
                if GPU:
                    next_histories = next_histories.to(device)
                    histories = histories.to(device)
                    y = y.to(device)
                future_Qs = target_dqn.forward(next_histories).detach()
                max_future_Qs = torch.max(future_Qs, 1)[0].view(BATCH_SIZE, 1)
                if GPU:
                    max_future_Qs = max_future_Qs.to(device)
                y += DISCOUNT_FACTOR * max_future_Qs
                # Calculate loss and perform gradient descent
                predicted_Qs = dqn.forward(histories).gather(1, torch.tensor([s[1] for s in sample], dtype=torch.long).view(BATCH_SIZE, 1))
                if GPU:
                    predicted_Qs = predicted_Qs.to(device)
                loss = loss_function(predicted_Qs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if done:
                    print(f'Episode: {episode_counter}')
                    print(f'Step: {step_counter}')
                    print(f'Total Reward: {total_reward}')
                    explore_probability = max(MINIMUM_EXPLORE_PROBABILITY,
                                              explore_probability - DELTA_EXPLORE_PROBABILITY)
                    print(f'Explore Probability: {explore_probability}')
                    break
            if done:
                break
            observation = new_observation

    torch.save(dqn, f'model-{epoch}')

env.close()