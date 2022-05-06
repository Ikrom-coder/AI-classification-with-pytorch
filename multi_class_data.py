import numpy as np
import os
import torch
from torch import nn
import torchvision
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.nn import functional as F

# 00000000000000000000000000000000000000000000000000000000000


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


def save_model():
    path = "myModel.pth"
    torch.save(model.state_dict(), path)
    print('Saved')

# 000000000000000000000000000000000000000000000000000000000000


train_data = os.path.join('multi-class_data/seg_train')
test_data = os.path.join('multi-class_data/seg_test')
image_data = os.path.join('multi-class_data/test')

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

transform = transforms.Compose([
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_p = transforms.Compose([
    transforms.CenterCrop(150),
    transforms.ToTensor(),
])
print('Preparing the data')


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
print('Preparing the Neural Network')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, (5, 5), padding=1)
        self.conv2 = nn.Conv2d(12, 24, (5, 5), padding=1)
        self.conv3 = nn.Conv2d(24, 48, (5, 5), padding=1)
        self.conv4 = nn.Conv2d(48, 48, (5, 5), padding=1)
        self.pool = nn.MaxPool2d(3, 3)
        self.dropout = nn.Dropout2d(0.4)
        self.batch_norm1 = nn.BatchNorm2d(12)
        self.batch_norm2 = nn.BatchNorm2d(24)
        self.batch_norm3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(8112, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 6)

    def forward(self, x):
        x = self.batch_norm1(F.relu(self.conv1(x)))
        x = self.batch_norm2(F.relu(self.conv2(x)))
        x = self.dropout(self.batch_norm2(self.pool(x)))
        x = self.batch_norm3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


model = Net()
# images, labels = next(iter(image))
# image_show(images, labels)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)


def test_accuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_image:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(image, 0):

            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = criterion(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()  # extract the loss value
            if i % 1000 == 999:
                # print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = test_accuracy()
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % accuracy)

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            save_model()
            best_accuracy = accuracy


# print('Training')
# train(5)
# print('Finished Training')

images, labels = next(iter(prediction_image))
# image_show(images, labels)
print('Real labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(6)))
path = "myModel.pth"
model1 = model.load_state_dict(torch.load(path))
outputs = model(images)

# We got the probability for every 6 labels. The highest (max) probability should be correct label
_, predicted = torch.max(outputs, 1)

# Let's show the predicted labels on the screen to compare with the real ones
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(6)))
