import numpy as np
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.nn import functional as F

shapes_data = os.path.join("shapes/train")
test_shapes_data = os.path.join("shapes/test")

classes = ['circle', 'square', 'star', 'triangle']

transform = transforms.Compose([
    transforms.CenterCrop(150),
    transforms.ToTensor(),
])


def jpg_to_png(source, new_format):
    from PIL import Image
    import os

    im1 = Image.open(source)
    im1.save(new_format)
    os.remove(source)
    print("Done!")


def image_show(image, labels_s):
    i = 1
    figure = plt.figure(figsize=(5, 5))
    for j in range(0, 4):
        figure.add_subplot(4, 4, i)
        plt.axis("off")
        img = image[j].squeeze()
        plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
        plt.title(classes[labels_s[j]])
        i += 1
    plt.show()


class CustomData(Dataset):
    def __init__(self, data, test_data, transform_s):
        self.data = data
        self.test_data = test_data
        self.transform = transform_s

    def __getitem__(self, item):
        pass

    def data_getter(self):
        train_set = datasets.ImageFolder(self.data, self.transform)
        data = DataLoader(train_set, batch_size=64, shuffle=True)
        return data

    def test_data_getter(self):
        test_set = datasets.ImageFolder(self.test_data, self.transform)
        t_data = DataLoader(test_set, batch_size=64, shuffle=True)
        return t_data


cds = CustomData(shapes_data, test_shapes_data, transform)
shapes_image = cds.data_getter()
test_shapes_image = cds.test_data_getter()

# source = os.path.join('shapes/test/star/star.jpg')
# new = os.path.join('shapes/test/star/star.png')
# jpg_to_png(source, new)

# images, labels = next(iter(test_shapes_image))
# image_show(images, labels)
# print(images[0].size())
# print(labels)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, (10, 10), padding=1)
        self.conv2 = nn.Conv2d(6, 16, (10, 10), padding=1)

        self.conv3 = nn.Conv2d(16, 32, (10, 10), padding=1)
        self.conv4 = nn.Conv2d(32, 64, (10, 10), padding=1)
        # self.conv5 = nn.Conv2d(64, 128, (5, 5))

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))

        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        # x = F.relu(self.pool(self.conv5(x)))

        x = x.view(in_size, -1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # return F.log_softmax(x, dim=1)

        return x


net = Net()


def train_ai(epoch_t, data_to_train):
    print('Training...')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epoch_t):
        running_loss = 0.0
        for i, data in enumerate(data_to_train, 0):
            inputs, label_s = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, label_s)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    print('Finished Training')


def accuracy_checker(test_set):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set:
            img, lbl = data
            outputs = net(img)
            _, predicted = torch.max(outputs.data, 1)
            total += lbl.size(0)
            correct += (predicted == lbl).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


train_ai(2, shapes_image)
# image_show(images, labels)
print('Saving the trained data ...')
PATH = './artificial.pth'
torch.save(net.state_dict(), PATH)
print('Saved')
accuracy_checker(test_shapes_image)

# images, labels = next(iter(test_shapes_image))
# image_show(images, labels)
# print('Classes: ',  ' '.join('%4s' % classes[labels[j]] for j in range(4)))
# net.load_state_dict(torch.load(PATH))
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join('%4s' % classes[predicted[j]] for j in range(4)))

