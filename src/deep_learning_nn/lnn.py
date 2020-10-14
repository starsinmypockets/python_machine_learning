import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

###
# TRAINING
###

# check balance of training data
total = 0
counter = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
  # xs is image matrix
  # ys is a tensor of the number values pictured
  xs, ys = data
  for y in ys:
    counter[int(y)] += 1
    total += 1

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    # images are 28x28 pixels
    # the linear neural network we are building is 64 layers
    self.fc1 = nn.Linear(28*28, 64)
    self.fc2 = nn.Linear(64, 64) # output of this layer is input of next
    self.fc3 = nn.Linear(64, 64)
    self.fc4 = nn.Linear(64, 10) # final output is number of bins (integer 0-9 we are classifying)

  def forward(self, x):
    x = F.relu(self.fc1(x)) # rectified linear activation function
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x) 

    return F.log_softmax(x, dim=1) # magic

net = Net()

X = torch.rand([28,28]).view(-1, 28*28)

# I think this is the gradient descent function
optimizer = optim.Adam(net.parameters(), lr = 1e-3)

# epochs are a pass through all of the training data
EPOCHS = 3

for epoch in range(EPOCHS):
  for data in trainset:
    # data is a batch of featuresets and labels
    X, y = data
    net.zero_grad() # zero the grient, not sure why this is
    output = net(X.view(-1, 28*28))
    loss = F.nll_loss(output, y) # loss calc -- this is often mean square error for vectors
    loss.backward()
    optimizer.step()
  print(loss)

###
# TESTING
###
correct = 0
total = 0

with torch.no_grad():
  for data in trainset:
    X, y = data
    output = net(X.view(-1, 28*28))
    for idx, n in enumerate(output):
      if torch.argmax(n) == y[idx]:
        correct += 1
      total += 1

print("Accurracy", round(correct/total, 3))
