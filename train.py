import pandas as pd
import numpy as np
import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torchvision.transforms as transforms
from model import LSTMEEGClasification
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
# from sklearn.model_selection import train_test_split #yerine random split li halini yazacağım.
from MyDataLoader import EEGClasificationDataLoader
import pdb 
plt.style.use('ggplot')
#validation ve traing  ve test için veri ayıracağım random split    +
#for döngüleri için için                                            +
#ilk eğitim hatası eğirisi çizdir                                   -
#haftasonun mail at

train_loss=["loss"]
validation_loss=["loss"]
test_loss=["loss"]
epochs = 50
batch_size = 8
lr = 0.0001


# model = model.LSTMEEGClasification().to(device)
model =LSTMEEGClasification()

optimizer = optim.Adam(model.parameters(), lr=lr)
#criterion = nn.BCELoss() # BCE los without sigmoid
criterion = nn.BCEWithLogitsLoss() # BCE los with sigmoid

# leanring parameters
epochsayısı = 50
batch_size = 8
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model=LSTMEEGClasification()


training_set=EEGClasificationDataLoader(secim=0)
print(training_set.shape)
training_generator = DataLoader(training_set,batch_size=8 )
# print(training_set)
validation_set=EEGClasificationDataLoader(secim=2)
validation_generator = torch.utils.data.DataLoader(validation_set,batch_size )
#Validation generator ve test generator
for epoch in range(epochsayısı):
        
        for (_,batch) in enumerate(training_generator):
            
            model=model.train()
            
            #modelin içine 4 tane değişkeni koyacapım  batch[0],.., batch[3]
            # print(batch[0].shape)
            
            output = model(batch[0],batch[1],batch[2],batch[3])
            # print("outputun shape'i ",len(output))b
            output_len=len(output)
            
            #outputun aynı şeklinde bir tensor tanımlayıp aynı olacak ve tüm değerleri 1 olacak
            # tensorum=torch.ones[]
            loss = criterion(output, torch.ones([output_len],dtype=torch.float32))
            
            # print("loss:",loss)
            
            #modelin içine son  4 tane değişkeni koyacapım  batch[4],.., batch[7]
            output_len=len(output)
            output = model(batch[4],batch[5],batch[6],batch[7])
            output_len=len(output)
            # outputun aynı şeklinde bir tensor tanımlayıp aynı olacak ve tüm değerleri 1 olacak
            loss += criterion(output, torch.zeros([output_len],dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print( "epoch:" ,epoch,  "loss:", loss.item() )
            
        # 10-20 li epochlarda validation hastasını hesapla ve hata eğrisini çizdir
        if epoch %10 ==0 and epoch !=0:
            # validation_set=EEGClasificationDataLoader(secim=2)
            # validation_generator = torch.utils.data.DataLoader(validation_set,batch_size )
            model=model.eval()   
            for (_,batch) in enumerate(validation_generator):
                
               
                #modelin içine 4 tane değişkeni koyacapım  batch[0],.., batch[3]
                output = model(batch[0],batch[1],batch[2],batch[3])
                loss = criterion(output, torch.ones([output_len*1],dtype=torch.float32))
        
                #modelin içine son  4 tane değişkeni koyacapım  batch[4],.., batch[7]
                output = model(batch[4],batch[5],batch[6],batch[7])
                loss += criterion(output, torch.zeros([output_len*0],dtype=torch.float32))
                
                
                print( "Validation epoch:", epoch, "batch:", batch, "loss:", loss.item() )
                
                if epoch==10:
                    validation_loss=[loss.item]
                else:
                    validation_loss.append(loss.item)
                #validation için  hata hesapla ve hata eğrisini çizdir
        if epoch==0:
            train_loss=[loss.items]
        else:
            train_loss.append(loss.item)
        
           
            
#test için sonuçları hesapla
test_set=EEGClasificationDataLoader(secim=1)
test_generator = torch.utils.data.DataLoader(test_set)
for epoch in range(epochsayısı):

        for batch in enumerate(test_generator):
            
            #modelin içine 4 tane değişkeni koyacapım  batch[0],.., batch[3]
            output = model(batch[0],batch[1],batch[2],batch[3])
            loss = criterion(output, torch.tensor([1]))
            
            #modelin içine son  4 tane değişkeni koyacapım  batch[4],.., batch[7]
            output = model(batch[4],batch[5],batch[6],batch[7])
            loss += criterion(output, torch.tensor([0]))
            
            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })          
            

# loss_train = history.history['train_loss']
# loss_val = history.history['val_loss']
# epochs = range(1,50)
# plt.plot(epochs, loss_train, 'g', label='Training loss')
# plt.plot(epochs, loss_val, 'b', label='validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
                
