import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, 4)                         # 4   x 84  x 84   ->  16  x 20  x 20
        self.conv2 = nn.Conv2d(16, 32, 4, 2)                        # 16  x 20  x 20   ->  32  x 9   x 9
        self.linear1 = nn.Linear(32 * 9 * 9, 256)                   # 32  x 9   x 9    ->  256
        self.linear2 = nn.Linear(256, 3)                            # 256              ->  3

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32*9*9)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x



