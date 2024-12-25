# Author    : zhiping
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from basic.common import *
from resnet import *

batch_size = 64
epoch = 10
number_work = 5

path = "../datasets"
train_set = datasets.CIFAR10(path, train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.CIFAR10(path, train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=number_work)
test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=number_work)

model = ResNet18()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

if __name__ == "__main__":
    for i in tqdm(range(epoch)):
        _model = train(train_loader, model, criterion, optimizer)
        vail(test_loader, _model)
