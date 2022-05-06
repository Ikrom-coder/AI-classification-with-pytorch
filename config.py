import torch
import matplotlib.pyplot as plt
import numpy as np
from neural_network import net
from torch import nn, optim


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


def image_show(image, labels_s):
    classes = ['Cat', 'Circle', 'Dog', 'Girl', 'Square', 'Star', 'Triangle']
    figure = plt.figure(figsize=(5, 5))
    for j in range(1, 10):
        figure.add_subplot(3, 3, j)
        plt.axis("off")
        img = image[j].squeeze()
        plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
        # plt.title(labels_s[j])
        for i in range(7):
            plt.title(classes[labels_s[j]])
    plt.show()


def train_ai(epoch_t, data_to_train):
    print('Training...')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

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
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')


def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch
    mean /= total_images_count
    std /= total_images_count

    return mean, std


def jpg_to_png(source, new_format):
    from PIL import Image
    import os

    im1 = Image.open(source)
    im1.save(new_format)
    os.remove(source)
