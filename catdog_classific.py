import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms, datasets
import torch

from neural_network import net
from config import train_ai, accuracy_checker, image_show


# test = os.path.join('Cat_and_Dog')

data_dir = os.path.join('Cat_Dog_data/train')
data_dir_test = os.path.join('Cat_Dog_data/test')

classes = ['Cat', 'Circle', 'Dog', 'Girl', 'Square', 'Star', 'Triangle']

mean = [0.4696, 0.4341, 0.3990]
std = [0.2468, 0.2370, 0.2354]

transform = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(100),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

print('Getting the data...')


class CustomDataSet(Dataset):
    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, data_s, data_test_s, batch_s, transform_s):
        super(Dataset, self).__init__()
        self.data = data_s
        self.data_t = data_test_s
        self.batch_size = batch_s
        self.transform = transform_s

    def getting_dog_and_cat_data(self):
        images_p = datasets.ImageFolder(self.data, self.transform)
        image_load = DataLoader(images_p, batch_size=self.batch_size, shuffle=True)
        return image_load

    def getting_test_dog_and_cat_data(self):
        test_dog_image = datasets.ImageFolder(self.data_t, self.transform)
        dog_test_load = DataLoader(test_dog_image, batch_size=self.batch_size, shuffle=True)
        return dog_test_load


cds = CustomDataSet(data_dir, data_dir_test, batch_s=400, transform_s=transform)
data_c_d = cds.getting_dog_and_cat_data()
test_data_c_d = cds.getting_test_dog_and_cat_data()
images, labels = next(iter(data_c_d))
print('Building ANN ...')
print('ANN Built')
print('Training init')
train_ai(2, data_c_d)
# image_show(images, labels)
print('Saving the trained data ...')
PATH = './artificial1.pth'
torch.save(net.state_dict(), PATH)
print('Saved')

# accuracy_checker(data_c_d)
# print(labels)
# image_show(images, labels)
# print('Classes: ',  ' '.join('%5s' % classes[labels[j]] for j in range(5)))
# net.load_state_dict(torch.load(PATH))
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(5)))
