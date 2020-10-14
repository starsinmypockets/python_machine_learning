import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    # convlutional layers
    self.conv1 == nn.Conv2d(1, 32, 5)
    self.conv2 == nn.Conv2d(32, 64, 5)
    self.conv3 == nn.Conv2d(64, 128, 5)

    hack = torch.rand(-1,-1,50,50)
    self._to_linear = None
    self.convs(x)

    # fully connected layers
    # self.fc1 = nn.Linear(,512)
    # self.fc2 = nn.Linear(512, 2)

  def convs(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
    x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
    x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))

    if self.to_linear is None:
      self.to_linear
    
