import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)                         # 4   x 84  x 84   ->  32  x 20  x 20
        self.conv2 = nn.Conv2d(32, 64, 4, 2)                        # 32  x 20  x 20   ->  64  x 9   x 9
        self.conv3 = nn.Conv2d(64, 64, 3, 1)                        # 64  x 9   x 9    ->  64  x 7   x 7
        self.linear1 = nn.Linear(64 * 7 * 7, 512)                   # 64  x 7   x 7    ->  512
        self.linear2 = nn.Linear(512, 3)                            # 512              ->  3

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x