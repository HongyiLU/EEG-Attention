from dataloader import create_dataset
from model.soft_attention import AutoEncoderRNN 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt

loss = []
epoch_count = []
loss_vals = []
acc_vals = []
val_loss_min = np.Inf
result_acc = []
result_loss = []


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(start_epoch, num_epochs, train_loader, val_loader, criterion, optimizer, device, model,
          val_loss_min):
    for epoch in range(start_epoch, num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        # train.
        model.train()
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels.long())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputs = outputs.argmax(dim=1)
            train_loss += loss.item()

        # evaluation.
        model.eval()
        for i, (data, labels) in enumerate(val_loader):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels.long())
            val_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1,
            train_loss,
            val_loss
        ))

    return model

def test(test_loader, criterion, device, model):
    confusion_pred = []
    confusion_act = []
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            confusion_pred += predicted.tolist()
            confusion_act += labels.tolist()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # avg_test_acc = sum(bal_acc_vals) / len(bal_acc_vals)
        # print('avg_loss and balanced_acc_scores', avg_test_acc)
    acc_avg = float(correct) * 100 / total
    class_type = ['0', '1', '2']
    confusion_mat = confusion_matrix(y_true=confusion_act, y_pred=confusion_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=class_type)
    disp = disp.plot()
    pre, rec, fscore, _ = score(y_true=confusion_act, y_pred=confusion_pred)
    print('test acc: {:.2f}%'.format(acc_avg))
    print('confusion_matrix \n', confusion_mat)
    print('fscores\n', fscore, pre, rec)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    train_data, val_data, test_data = create_dataset()
    train_loader = DataLoader(train_data, shuffle=True, batch_size=256, drop_last=True)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=256, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=256, drop_last=True)
    model = AutoEncoderRNN(input_size = 7,input_size_dec=512, hidden_size = 512, num_layers = 4, sequence_length=20, output_size = 3, dropout = 0.2).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    trained_model = train(0, 1, train_loader, val_loader, criterion, optimizer,
                          device,
                          model, val_loss_min)
    trained_model.eval()
    test(test_loader, criterion, device, model)

