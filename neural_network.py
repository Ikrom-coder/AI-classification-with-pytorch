from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (5, 5), padding=1)
        self.conv2 = nn.Conv2d(6, 16, (5, 5), padding=1)
        self.conv3 = nn.Conv2d(16, 32, (5, 5), padding=1)
        self.conv4 = nn.Conv2d(32, 64, (5, 5))
        self.conv5 = nn.Conv2d(64, 128, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2048, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 7)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        x = F.relu(self.pool(self.conv5(x)))

        x = x.view(in_size, -1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # return F.log_softmax(x, dim=1)

        return x


net = Net()
