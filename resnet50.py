import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')


# show image
def image_show(image_s, labels_s):
    m = 1
    figure = plt.figure(figsize=(5, 5))
    for j in range(6):
        figure.add_subplot(3, 3, m)
        plt.axis("off")
        img = image_s[j].squeeze()
        plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
        plt.title(classes[labels_s[j]])
        m += 1
    plt.show()


# accuracy
def accuracy_check(data_image):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images_1, labels_1 in data_image:
            images_1 = images_1.to(device)
            labels_1 = labels_1.to(device)
            outputs_1 = model(images_1)
            _, predicted = torch.max(outputs_1.data, 1)
            total += labels_1.size(0)
            correct += (predicted == labels_1).sum().item()
    acc = 100 * correct / total
    return acc


# locc check
def loss_checker(img, lbl):
    running_loss = 0.0
    img, lbl = img.to(device), lbl.to(device)
    out = model(img)
    loss_1 = criterion(out, lbl)
    optimizer.zero_grad()
    loss_1.backward()
    optimizer.step()
    running_loss += loss_1.item()
    return running_loss


# Hyper-parameters
accuracy_train, accuracy_test, loss_train, loss_test = [], [], [], []
batch_size = 128
learning_rate = 0.001


# Image preprocessing modules
transform = transforms.Compose([
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_p = transforms.Compose([
    transforms.CenterCrop(150),
    transforms.ToTensor(),
])

# real images dataset
train_data = os.path.join('multi-class_data/seg_train')
test_data = os.path.join('multi-class_data/seg_test')
image_data = os.path.join('multi-class_data/test')

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


class CustomData(Dataset):
    def __init__(self, t_data, te_data, p_data, transform_s, transforms_p):
        self.train_data = t_data
        self.test_data = te_data
        self.predicted_data = p_data
        self.transform = transform_s
        self.transform_p = transforms_p

    def __getitem__(self, item):
        pass

    def data_getter(self):
        train_set = datasets.ImageFolder(self.train_data, self.transform)
        data = DataLoader(train_set, batch_size=128, shuffle=True)
        return data

    def test_data_getter(self):
        test_set = datasets.ImageFolder(self.test_data, self.transform)
        t_data = DataLoader(test_set, batch_size=128, shuffle=True)
        return t_data

    def prediction_data_getter(self):
        pred_set = datasets.ImageFolder(self.predicted_data, self.transform_p)
        p_data = DataLoader(pred_set, batch_size=10, shuffle=True)
        return p_data


cds = CustomData(train_data, test_data, image_data, transform, transform_p)
image = cds.data_getter()
test_image = cds.test_data_getter()
prediction_image = cds.prediction_data_getter()


# 3x3 convolution
print('Preparing the Neural Network')


def conv3x3(in_channels, out_channels, stride=(1, 1)):
    return nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=6):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(1024, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for m in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate
def update_lr(optimizer1, lr):
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr


# Train the model
def train(num_epochs):
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(image):
            # images = images.to(device)
            # labels = labels.to(device)
            #
            # # Forward pass
            # outputs = model(images)
            # loss = criterion(outputs, labels)
            #
            # # Backward and optimize
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            loss_train.append(loss_checker(images, labels))
            accuracy_train.append(accuracy_check(image))
            if (i + 1) % 100 == 0:
                print(f"(train)Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_checker(images, labels)}")
                print(f'Accuracy of the model on the images: {accuracy_check(image)}')
        for m, (images1, labels1) in enumerate(test_image):

            loss_test.append(loss_checker(images1, labels1))
            accuracy_test.append(accuracy_check(test_image))
            if (m + 1) % 100 == 0:
                print(f"(test)Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_checker(images1, labels1)}")
                print(f'Accuracy of the model on the test images: {accuracy_check(test_image)}')

        # Decay learning rate
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)


print('Training')
train(50)
print('Finished Training')
print('Saving')
torch.save(model.state_dict(), 'myResnet50.pth')
print('Saved')

print(loss_train, loss_test)
plt.plot(loss_train, label='Training loss')
plt.plot(loss_test, label='Test loss')
plt.legend(frameon=False)
plt.show()

print(accuracy_train, accuracy_test)
plt.plot(accuracy_train, label='Train Accuracy')
plt.plot(accuracy_test, label='Test Accuracy')
plt.legend(frameon=False)
plt.show()

