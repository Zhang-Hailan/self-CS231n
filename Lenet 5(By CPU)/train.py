#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# transform为预处理函数
# ToTensor作用
# ![image.png](attachment:image.png)
# Normalize作用
# ![image-2.png](attachment:image-2.png)

# In[2]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# 第一次设置的时候将download改为True 会在本地文件夹下面下载数据集

# In[3]:


train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=False, transform=transform)


# 每一批次随机拿出36个数据集

# In[4]:


train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                           shuffle=True, num_workers=0)


# In[5]:


# 10000张验证图片
val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                         shuffle=False, num_workers=0)


# 迭代器

# In[6]:


val_data_iter = iter(val_loader)
val_image, val_label = next(val_data_iter)


# index 0==plane 1==car ..

# In[7]:


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 可以看看导入的图像 将size改为较小的数字4 可以看到图像的显示

# In[8]:


# 将处理的图像返回出去 用来显示 包括大小 各种通道数
def imshow(img):
    img = img / 2 + 0.5  # 反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
print("".join("%5s" % classes[val_label[j]] for j in range(4)))
imshow(torchvision.utils.make_grid(val_image))


# ![image.png](attachment:image.png)

# In[9]:


net = LeNet()  # 实例化模型
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 使用Adam优化器
loss_function = nn.CrossEntropyLoss()  # 已经包含了softmax


# In[11]:


for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        # step返回的就是索引index   get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # 不清除历史梯度 便会对历史梯度进行累加 通过这个特性便可以实现一个很大batch数值的训练
        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

         # print statistics
        running_loss += loss.item()
        if step % 500 == 499:    # print every 500 mini-batches
            with torch.no_grad():   # 接下来的过程中不需要计算损失误差梯度 占用更多的资源 和 占用更多的内存资源
                outputs = net(val_image)  # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0
print('Finished Training')
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)


# 训练好的参数保存在当前路径下的Lenet.pth文件夹里面
# 后续可以在predict文件夹里面对很多图像进行训练
