import math
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.dataset import T_co
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np


# # real images dataset
train_data = os.path.join('multi-class_data/seg_train')
test_data = os.path.join('multi-class_data/seg_test')
image_data = os.path.join('multi-class_data/test')

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
print('Customizing the data')

x = 224
y = 224

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Resize((x, y)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_p = transforms.Compose([
    transforms.Resize((x, y)),
    transforms.ToTensor(),
])
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')


class CustomData(Dataset):
    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, t_data, te_data, p_data, transform_s, transforms_p):
        self.train_data = t_data
        self.test_data = te_data
        self.predicted_data = p_data
        self.transform = transform_s
        self.transform_p = transforms_p

    def data_getter(self):
        train_set = datasets.ImageFolder(self.train_data, self.transform)
        data_g = DataLoader(train_set, batch_size=50, shuffle=True)
        return data_g, len(train_set)

    def test_data_getter(self):
        test_set = datasets.ImageFolder(self.test_data, self.transform)
        t_data = DataLoader(test_set, batch_size=50, shuffle=True)
        return t_data, len(test_set)

    def prediction_data_getter(self):
        pred_set = datasets.ImageFolder(self.predicted_data, self.transform_p)
        p_data = DataLoader(pred_set, batch_size=10, shuffle=True)
        return p_data


cds = CustomData(train_data, test_data, image_data, transform, transform_p)
image, trained_len = cds.data_getter()
val_image, tested_len = cds.test_data_getter()
pred_image = cds.prediction_data_getter()
len_data = {
    'train': trained_len,
    'val': tested_len
}
data = {
    'train': image,
    'val': val_image,
}

# print('Preparing the Neural Network')


def conv3x3(in_planes, out_planes, stride=(1, 1)):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x2):
        residual = x2

        out = self.conv1(x2)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x2)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x3):
        residual = x3

        out = self.conv1(x3)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x3)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_classes):
        super(ResNet, self).__init__()
        blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3],
                  200: [3, 24, 36, 3]}
        assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=(1, 1)):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, out):
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model_res = ResNet(depth=50, num_classes=6).to(device)


def image_show(image_s, labels_s):
    i = 1
    figure = plt.figure(figsize=(5, 5))
    for j in range(6):
        figure.add_subplot(3, 3, i)
        plt.axis("off")
        img = image_s[j].squeeze()
        plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
        plt.title(classes[labels_s[j]])
        i += 1
    plt.show()

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss().cuda()
# optimizer = torch.optim.SGD(model_res.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
# tr_acc, te_acc, tr_loss, te_loss = [], [], [], []
#
#
# def train_model(model, criterion, optimizer, num_epochs=50):
#     best_accuracy = 0.0
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch+1, num_epochs))
#         print('-' * 30)
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             for inputs, labels in data[phase]:
#                 inputs = inputs.cuda()
#                 labels = labels.cuda()
#
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#
#                 if phase == 'train':
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#
#                 _, preds = torch.max(outputs, 1)
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#
#             epoch_loss = running_loss / len_data[phase]
#             epoch_acc = running_corrects / len_data[phase]
#             if phase == 'train':
#                 tr_acc.append(epoch_acc.cpu().detach().numpy())
#                 tr_loss.append(epoch_loss)
#             elif phase == 'val':
#                 te_acc.append(epoch_acc.cpu().detach().numpy())
#                 te_loss.append(epoch_loss)
#             print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, 100*epoch_acc))
#             if phase == 'train':
#                 if epoch_acc > best_accuracy:
#                     path = "myResnet50.pth"
#                     torch.save(model.state_dict(), path)
#                     # print('Saved')
#                     best_accuracy = epoch_acc
#
#
# train_model(model_res, criterion, optimizer)

# tr_loss = [2.8522, 1.1481, 0.9313, 0.8290, 0.7582, 0.7090,
#            0.6538, 0.6146, 0.5803, 0.5545, 0.5149, 0.4963,
#            0.4648, 0.4512, 0.4262, 0.3993, 0.3886, 0.3710, 0.3554, 0.3339, 0.3166, 0.3052, 0.2894, 0.2627, 0.2500]
# te_loss = [1.4058, 1.1607, 0.9624, 0.7777, 0.6948, 0.6715,
#            0.6988, 1.2311, 0.6400, 0.5136, 0.4946, 0.5126,
#            0.4792, 0.4902, 0.4333, 0.4768, 0.5870, 0.5230, 0.4571, 0.4821, 0.4323, 0.5568, 0.5358, 0.4820, 0.5279]
# tr_acc = [36.6966, 52.7362, 63.1823, 67.6927, 71.0560, 73.1438,
#           75.5451, 77.0628, 78.6091, 79.6280, 81.5021, 82.1362,
#           83.1124, 83.4830, 84.6302, 85.8415, 86.2121, 86.7180,
#           87.0030, 87.8581, 88.4566, 89.2475, 89.4257, 90.6940, 90.9862]
# te_acc = [38.7667, 54.5000, 60.2667, 71.0000, 73.9667, 73.8000,
#           73.1667, 54.8000, 75.7667, 81.4333, 82.5333, 81.8333,
#           83.6333, 83.0667, 85.1000, 83.1000, 79.6000, 81.8000,
#           83.9333, 82.2667, 84.5000, 81.2000, 82.1000, 83.5000, 83.9333]

# plt.figure(figsize=(10, 5))
# plt.title("Training and Validation Loss")
# plt.plot(te_loss, label="val")
# plt.plot(tr_loss, label="train")
# plt.xlabel("Validation")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(10, 5))
# plt.title("Training and Validation Accuracy")
# plt.plot(te_acc, label="val")
# plt.plot(tr_acc, label="train")
# plt.xlabel("Validation")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

images, labels = next(iter(pred_image))
print('Real labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(6)))
path = "myResnet50.pth"
model_res.load_state_dict(torch.load(path, map_location='cpu'))
outputs = model_res(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(6)))
image_show(images, predicted)
