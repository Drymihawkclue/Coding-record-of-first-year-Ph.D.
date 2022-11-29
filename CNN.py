# import libraries
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data as data
from utils import *

# define settings
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=50,
                    help='number of classes used')
parser.add_argument('--num_samples_train', type=int, default=15,
                    help='number of samples per class used for training')
parser.add_argument('--num_samples_test', type=int, default=5,
                    help='number of samples per class used for testing')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
args = parser.parse_args()

# define you model, loss functions, hyperparameters, and optimizers
### Your Code Here ###
class covNet(nn.Module):
    def __init__(self):
        super(covNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1),nn.BatchNorm2d(32),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3136, 512)
        self.BN = nn.BatchNorm2d(64)
        self.BN1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(512, 50)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out=self.BN(out)
        out = out.view(in_size, -1)
        out = self.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        return self.logsoftmax(out)


model=covNet()
print(model)
# LOSS
LOSS_FUN= torch.nn.CrossEntropyLoss()
# Optimizer
opt=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.01)
epochs = 200

# load data
train_image, train_label, test_image, test_label = LoadData(args.num_classes, args.num_samples_train,
                                                            args.num_samples_test, args.seed)
# note: you should use train_image, train_label for training, apply the model to test_image to get predictions and use test_label to evaluate the predictions
train_image = train_image.reshape(-1,1,28,28)
test_image = test_image.reshape(-1,1,28,28)
train_image = torch.from_numpy(train_image.astype(np.float32))
test_image = torch.from_numpy(test_image.astype(np.float32))
train_label = torch.from_numpy(train_label)
train_label=train_label.long()
print(train_label)
test_label = torch.from_numpy(test_label)
test_label=test_label.long()
print(test_label)
# train model using train_image and train_label
loss_list = []
acc_list = []
for epoch in range(epochs):
    model.train()
    opt.zero_grad()
    ### Your Code Here ###
    output=model(train_image)
    loss = LOSS_FUN(output,train_label)
    loss.backward()
    opt.step()
    print(loss)
    loss_list.append(loss.item())
    pred1 = output.data.max(1)[1]
    acc = torch.mean(1.0 * (pred1 == train_label))
    acc_list.append(acc.item())
    #print(loss_list)

x1=range(0,epochs)
plt.plot(x1,loss_list,'o-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.savefig('loss.jpg')
plt.plot(x1,acc_list,'o-')
plt.xlabel('Epoch')
plt.ylabel('Train accuracy')
plt.show()
plt.savefig('acc.jpg')

print("Train Accuracy:", torch.mean(1.0 * (pred1 == train_label)))

# get predictions on test_image
model.eval()
with torch.no_grad():
    ### Your Code Here ###
    out=model(test_image)
    pred = out.data.max(1)[1]

# evaluation
print("Test Accuracy:", torch.mean(1.0 * (pred == test_label)))
# note that you should not use test_label elsewhere





