import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Net import Net
import cv2

batch_size = 30

# 下载训练集
# train_dataset = datasets.MNIST(root='./num/',
#                                train=True,
#                                transform=transforms.ToTensor(),
#                                download=True)
train_dataset = datasets.ImageFolder(root='/Users/liuyiqi/Nutstore Files/同步文档/2020第一学期课程/智能系统/CNN/train',
                                     transform=transforms.Compose([
                                         transforms.CenterCrop(28),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.Grayscale(),
                                         transforms.ToTensor(),
                                     ]))
print(train_dataset.class_to_idx)

# 下载测试集
# test_dataset = datasets.MNIST(root='./num/',
#                               train=False,
#                               transform=transforms.ToTensor(),
#                               download=True)
test_dataset = datasets.ImageFolder(root='/Users/liuyiqi/Nutstore Files/同步文档/2020第一学期课程/智能系统/CNN/test',
                                    transform=transforms.Compose([
                                        transforms.CenterCrop(28),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                    ]))
print(test_dataset.class_to_idx)

# dataset 参数用于指定我们载入的数据集名称
# batch_size参数设置了每个包中的图片数据个数
# 在装载的过程会将数据随机打乱顺序并进打包

# 建立一个数据迭代器
# 装载训练集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# 装载测试集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

# 实现单张图片可视化
# images, labels = next(iter(train_loader))
# img = torchvision.utils.make_grid(images)
#
# img = img.numpy().transpose(1, 2, 0)
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img * std + mean
# print(labels)
# cv2.imshow('win', img)
# key_pressed = cv2.waitKey(0)

# 训练
device = torch.device('cpu')
LR = 0.001

net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

epoch = 7
for epoch in range(epoch):
    sum_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()  # 梯度归零
        outputs = net(inputs)  # 前向运算
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        print(loss.item())

# 测试
net.eval()  # 转为测试模式
correct = 0
total = 0
for data_test in test_loader:
    images, labels = data_test
    images, labels = Variable(images), Variable(labels)
    output_test = net(images)
    _, predicted = torch.max(output_test, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print("correct1: ", correct)
print("test acc: {0}".format(correct.item() / len(test_dataset)))
