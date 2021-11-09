#modeli oluşturacağım yer
import torch
import torchvision
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import DataLoader
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F
from MyDataLoader import EEGClasificationDataLoader
import pdb


# import DataLoader

# features = 16
# # Hyperparameters
# input_size = 4
# hidden_size = 256
# num_layers = 5
# num_classes = 10
# sequence_length = 28
# learning_rate = 0.005
# batch_size = 8
# num_epochs = 2
# embedding_dim=256

class LSTMEEGClasification(nn.Module):
    def __init__(self):
        super(LSTMEEGClasification, self).__init__()
        
        self.l1 = nn.LSTM(input_size=256, hidden_size=128,num_layers=1,bidirectional=True)
        
        self.l2 = nn.LSTM(input_size=128, hidden_size=64,num_layers=1,bidirectional=True)
        
        self.l3 = nn.LSTM(input_size=64, hidden_size=32,num_layers=1,bidirectional=True)
        
        self.l4 = nn.LSTM(input_size=64, hidden_size=32,num_layers=1,bidirectional=True)
        #torchcat ile bu 4 lstm katmanını birleşireceğim ve çıkışı alacağım 
      
        self.l5 = nn.Linear(512,1 ,bias=True)
          # 4 tane lstm katmanı 1 lineer katmandan oluşacak modelim
        print("Model çalışıyor")
 
     

    def forward(self, CH,CHL,CLLL,CHLL):
        # pdb.set_trace()
        cho=self.l1(CH)[0]
        # print("cho.shape", cho.shape)
        chlo=self.l2(CHL)[0] 
        clllo=self.l3(CLLL)[0]
        chllo=self.l4(CHLL)[0]
        
        toplama=torch.cat((cho,chlo,clllo,chllo),2)
    
        toplama =self.l5(toplama)
        #sigmoid kullanalım
        toplama=F.log_softmax(toplama)
        return torch.squeeze(toplama) 

     