from pickletools import optimize
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from model.vit_mnist import ViT
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#input size
image_height = 28 
image_width = 28
#seperated patch size
patch_height = 7 
patch_width = 7
#num of classes
num_classes = 10 
#dim of linear layer
dim = 128 
#num of transformer block
depth = 6 
#num of heads
heads = 8 
#dim of mlp for classifing
mlp_dim = 128 
#input channels
channels = 1 
#multi head attention dimension
dim_head = 64 

dropout = 0.
emb_dropout = 0.
num_epochs = 10
lr = 0.001

loss_his = []

#Train fucntion
def train(num_epochs, train_loader, criterion, optimizer, model):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            y_out = model(data)
            loss = criterion(y_out, labels) / len(train_loader)
        
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_out = y_out.argmax(dim=1)
            train_loss += loss.item()
        
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch + 1,
            train_loss
        ))

        loss_his.append(train_loss)

    return model

#Evaluate fucntion
def evaluate(test_loader, criterion, model):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    for i, (data, labels) in enumerate(test_loader):
            y_out = model(data)
            loss = criterion(y_out, labels) / len(test_loader)
            test_loss += loss.item()

            _, predicted = torch.max(y_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc_avg = float(correct) * 100 / total
    print('Test acc: {:.2f}% \tTest loss : {:.6f}'.format(acc_avg, test_loss))

if __name__ == '__main__':
    transform = ToTensor()

    train_set = MNIST(root='./MNIST Data', train=True, download=True, transform=transform)
    test_set = MNIST(root='./MNIST Data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=16)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=16)
    #model
    model = ViT(image_height, image_width, patch_height, patch_width, num_classes, dim, depth, heads, mlp_dim, channels, dim_head, dropout, emb_dropout)
    optimizer = Adam(model.parameters(), lr)
    criterion = CrossEntropyLoss()
    model_trained = train(num_epochs, train_loader, criterion, optimizer, model)
    evaluate(test_loader, criterion, model_trained)
    plt.plot(range(num_epochs), loss_his)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()