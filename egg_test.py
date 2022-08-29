import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dataloader import create_dataset
from model.vit_mnist import ViT
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

#input size
image_height = 36 
image_width = 14
#seperated patch size
patch_height = 4
patch_width = 7
#num of classes
num_classes = 3 
#dim of linear layer
dim = 128
#num of transformer block
depth = 4       
#num of heads
heads = 8      
#dim of mlp for classifing
mlp_dim = 32      
#multi head attention dimension
head_dim = 64 
batch_size = 64   
dropout = 0.3            
emb_dropout = 0.0 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH = 'trainings_RoF/Epochs[200]_Lr[0.00013]_Dropout[0.3]_Batch[64]_Dim[128]_Depth[4]_Heads[8]_HDim[64]_MlpDim[32]_Time[29-15-39]_bfn'

def load_best_ckp(model):
    checkpoint = torch.load('./{}/best_checkpoint/best_model.pt'.format(PATH))
    model.load_state_dict(checkpoint['state_dir'])
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
    disp.plot()
    plt.plot()
    pre, rec, fscore, _ = score(y_true=confusion_act, y_pred=confusion_pred)
    print('confusion_matrix \n', confusion_mat)
    print('fscores\n', fscore, pre, rec)
    
if __name__ == '__main__':
    train_data, val_data, test_data = create_dataset()
    test_loader = DataLoader(test_data, shuffle=True, batch_size = batch_size)
    #model
    model = ViT(image_height, image_width, patch_height, patch_width, num_classes, dim, depth, heads, mlp_dim, head_dim, dropout, emb_dropout)
    model = load_best_ckp(model)
    model = model.to(device)
    criterion = CrossEntropyLoss()
    evaluate(test_loader, criterion, model)