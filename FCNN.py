# import libraries
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
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
class myNet(nn.Module):
    def __init__(self):
        super(myNet,self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(28*28,1000), nn.BatchNorm1d(1000), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1000,4000), nn.BatchNorm1d(4000), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(4000, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(512, 50))
        self.BN = nn.BatchNorm1d(784)
        self.dropout = nn.Dropout(p=0.9)
        self.logsoftmax = nn.LogSoftmax()
    def forward(self,x):
        out = self.BN(x)
        out = self.dropout(self.fc1(out))
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return self.logsoftmax(out)


model=myNet()
print(model)
#LOSS
LOSS_FUN= torch.nn.CrossEntropyLoss()
#Optimizer
opt=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.01)
epochs = 100

# load data
train_image, train_label, test_image, test_label = LoadData(args.num_classes, args.num_samples_train,
                                                            args.num_samples_test, args.seed)
# note: you should use train_image, train_label for training, apply the model to test_image to get predictions and use test_label to evaluate the predictions
train_image = torch.from_numpy(train_image.astype(np.float32))
test_image = torch.from_numpy(test_image.astype(np.float32))
train_label = torch.from_numpy(train_label)
train_label=train_label.long()
print(train_label)
test_label = torch.from_numpy(test_label)
test_label=test_label.long()
print(test_label)

# train model using train_image and train_label
for epoch in range(epochs):
    model.train()
    opt.zero_grad()
    ### Your Code Here ###
    output = model(train_image)
    loss = LOSS_FUN(output, train_label)
    print(loss)
    loss.backward()
    opt.step()
pred1 = output.data.max(1)[1]
print("Train Accuracy:", torch.mean(1.0 * (pred1 == train_label)))

# get predictions on test_image
model.eval()
with torch.no_grad():
    ### Your Code Here ###
    out = model(test_image)
    pred = out.data.max(1)[1]

# evaluation
print("Test Accuracy:", torch.mean(1.0 * (pred == test_label)))
# note that you should not use test_label elsewhere





