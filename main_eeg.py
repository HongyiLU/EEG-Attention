from cProfile import label
import os
from pickletools import optimize
from turtle import color
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataloader import create_dataset

from model.vit_mnist import ViT
import matplotlib.pyplot as plt
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#input size
image_height = 16 
image_width = 14
#seperated patch size
patch_height = 8 
patch_width = 7
#num of classes
num_classes = 3 
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
head_dim = 64 
batch_size = 128
dropout = 0.
emb_dropout = 0.
num_epochs = 50
lr = 0.0001

model_name = 'Epochs[{}]_Lr[{}]_Dropout[{}]_Batch[{}]_Dim[{}]_Depth[{}]_Heads[{}]_HDim[{}]_MlpDim[{}]'.format(
    num_epochs, lr, dropout, batch_size, dim, depth, heads, head_dim, mlp_dim)
working_dir = './trainings_RoF/' + model_name
best_model_dir = working_dir + '/best_checkpoint/'
best_model_path = best_model_dir + '/best_model.pt'

train_loss_his = []
val_loss_his = []
#Train fucntion
def train(num_epochs, train_loader, val_loader, criterion, optimizer, model):
    model.train()
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        val_true = 0
        train_loop = tqdm(train_loader, unit='batch', total=len(train_loader))
        
        for i, (data, labels) in enumerate(train_loop):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
        
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            train_loop.set_postfix(training_loss = '{:.6f}'.format(train_loss / len(train_loader)))
            
        # validation
        model.eval()
        val_loop = tqdm(val_loader, unit='batch', total=len(val_loader))
        for i, (data, labels) in enumerate(val_loop):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels.long())
            _, predicted = torch.max(outputs.data, 1)
            val_loss += loss.item()
            val_true += (predicted == labels).sum()
            val_acc = float(val_true)/float(batch_size*i+len(data))
            val_loop.set_description(f'Validation [{epoch + 1}/{num_epochs}]')
            val_loop.set_postfix(validation_loss = '{:.6f}'.format(val_loss / len(val_loader)), validation_acc = val_acc)

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        train_loss_his.append(train_loss)
        val_loss_his.append(val_loss)
    return model

#Evaluate fucntion
def evaluate(test_loader, criterion, model):
    confusion_pred = []
    confusion_act = []
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    for i, (data, labels) in enumerate(test_loader):
        data = data.to(device)
        labels = labels.to(device)
        y_out = model(data)
        loss = criterion(y_out, labels)
        test_loss += loss.item()
            
        _, predicted = torch.max(y_out.data, 1)
        confusion_pred += predicted.tolist()
        confusion_act += labels.tolist()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_loss = test_loss / len(test_loader)
    acc_avg = float(correct) * 100 / total
    print('Test acc: {:.2f}% \tTest loss : {:.6f}'.format(acc_avg, test_loss))
    class_type = ['0', '1', '2']
    confusion_mat = confusion_matrix(y_true=confusion_act, y_pred=confusion_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=class_type)
    disp = disp.plot()
    plt.savefig(working_dir + '/confusion_matrix.png')
    pre, rec, fscore, _ = score(y_true=confusion_act, y_pred=confusion_pred)
    print('confusion_matrix \n', confusion_mat)
    print('fscores\n', fscore, pre, rec)
    
    f = open(working_dir + '/res_evaluate.txt', 'w')
    f.write('test acc: {:.2f}%'.format(acc_avg) + '\n')
    f.write('tconfusion_matrix \n:' + np.array2string(confusion_mat) + '\n')
    f.write('fscores\n' + np.array2string(fscore) + np.array2string(pre) + np.array2string(rec))
    f.close()
    

def plot():
    plt.figure()
    plt.plot(range(num_epochs), train_loss_his, color='b', label = 'train loss')
    plt.plot(range(num_epochs), val_loss_his, color='r', label = 'val loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(working_dir + '/loss.png')

if __name__ == '__main__':
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
        os.makedirs(best_model_dir)
    else:
        print('Training exists: {}'.format(working_dir))
        raise SystemExit(1)

    train_data, val_data, test_data = create_dataset()
    train_loader = DataLoader(train_data, shuffle=False, batch_size = batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size = batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size = batch_size)
    #model
    model = ViT(image_height, image_width, patch_height, patch_width, num_classes, dim, depth, heads, mlp_dim, channels, head_dim, dropout, emb_dropout)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr)
    criterion = CrossEntropyLoss()
    model_trained = train(num_epochs, train_loader, val_loader, criterion, optimizer, model)
    plot()
    evaluate(test_loader, criterion, model_trained)
    