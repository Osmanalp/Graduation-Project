import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import preprocessing

import scipy.io
import torch.utils
import pywt
# import VeriSetiHazırlama

class EEGClasificationDataLoader(torch.utils.data.Dataset):

   

    def __init__(self,secim):# buraya labels eklenecek
        #datayı bölütle train / validation / test  
        # self.clear_data = "data/cdata/EEG_all_epochs.npy"
        # self.noisy_data = "data/ndata/noisyndata.npy"
        # self.traindata = VeriSetiHazırlama.X_train
        # self.validationdata = VeriSetiHazırlama.X_val
        # self.testdata = VeriSetiHazırlama.testdata
        self.traindata = "X_train.npy"
        self.validationdata = "X_test.npy"
        self.testdata = "X_test.npy"
        self.y_train="y_train.npy"
        self.y_test="y_test.npy"
        self.secim=secim
        # self.labels=labels
        
        # print(self.traindata.shape)
        print("DataLoader Çalışıyor")
        
        # print("data seçiminiz{}".format(secim))

    
    def __len__(self):
        #ifli secim yapacağım ona göre data verecek
        # secim=self.secim(int(input("train / test val için seçimi giriniz")))
        secim=self.secim
        if secim == 0:
            a=np.load(self.traindata)
            print("veri  tipi: ",type(a))
            traindata_arr=np.load(self.traindata).shape[0]
            trainlabel_arr=np.load(self.y_train).shape[0]
            print(type(traindata_arr))
            print(type(trainlabel_arr))
            #train için dataset seçilir
            return traindata_arr,trainlabel_arr
        
        elif secim == 1:
            #test için dataset seçilir
            return np.load(self.testdata).shape[0],np.load(self.y_test).shape[0]
        
        elif secim == 2:
            # print(np.load(self.validationdata).shape[0])
            # validation için dataset seçilir
            return np.load(self.validationdata).shape[0],np.load(self.y_test).shape[0]
        

    
    # def TrainDatagetitem(self,traindata):
    #     # clearalldata = np.load(self.traindata)[idx,:]
    #     x =traindata
    #     clearnormalized = (x-min(x))/(max(x)-min(x)+0.0001)
    #     #dalgacık dönüşümü ıuyguluyoruz her 2 sinyalede
    #     coeffs2 = pywt.dwt(clearnormalized, 'db1') #ilk dalgacık dönüşümü 
    #     CL, CH= coeffs2
    #     coeffs2 = pywt.dwt(CL, 'db1')#2. dalgacık dönüşümü 
    #     CLL, CHL= coeffs2
    #     coeffs2 = pywt.dwt(CLL, 'db1')#3. dalgacık dönüşümü 
    #     CLLL, CHLL= coeffs2
    #     return CH, CHL,CLLL,CHLL
    # def ValidationDatagetitem(self,validationdata):
    #     z=validationdata
    #     valnormalized = (z-min(z))/(max(z)-min(z)+0.0001)
    #     coeffs2 = pywt.dwt(valnormalized, 'db1') #ilk dalgacık dönüşümü 
    #     NorL, NorH= coeffs2
    #     coeffs2 = pywt.dwt(NorL, 'db1')#2. dalgacık dönüşümü 
    #     NorLL, NorHL= coeffs2
    #     coeffs2 = pywt.dwt(NorLL, 'db1')#3. dalgacık dönüşümü 
    #     NorLLL, NorHLL= coeffs2
    #     return NorH,NorHL,NorLLL,NorHLL
    # def TestDatagetitem(self,noisyalldata):
    #     y = noisyalldata
    #     noisynormalized = (y-min(y))/(max(y)-min(y)+0.0001)
    #     coeffs2 = pywt.dwt(noisynormalized, 'db1') #ilk dalgacık dönüşümü 
    #     NL, NH= coeffs2
    #     coeffs2 = pywt.dwt(NL, 'db1')#2. dalgacık dönüşümü 
    #     NLL, NHL= coeffs2
    #     coeffs2 = pywt.dwt(NLL, 'db1')#3. dalgacık dönüşümü 
    #     NLLL, NHLL= coeffs2
    #     return NH,NHL,NLLL,NHLL
        
    def __getitem__(self, idx):
       
        
        if self.secim==0:
            clearalldata = np.load(self.traindata)[idx,:]
            
            x =clearalldata
            clearnormalized = (x-min(x))/(max(x)-min(x)+0.0001)
            #dalgacık dönüşümü ıuyguluyoruz her 2 sinyalede
            coeffs2 = pywt.dwt(clearnormalized, 'db1') #ilk dalgacık dönüşümü 
            CL, CH= coeffs2
            coeffs2 = pywt.dwt(CL, 'db1')#2. dalgacık dönüşümü 
            CLL, CHL= coeffs2
            coeffs2 = pywt.dwt(CLL, 'db1')#3. dalgacık dönüşümü 
            CLLL, CHLL= coeffs2
            print(CH.shape)
            print(type(CH))
            return CH, CHL,CLLL,CHLL
            
            # CH, CHL,CLLL,CHLL=TrainDatagetitem(clearalldata)
        elif self.secim==2:
            validationdata=np.load(self.validationdata)[idx,:]
            z=validationdata
            valnormalized = (z-min(z))/(max(z)-min(z)+0.0001)
            coeffs2 = pywt.dwt(valnormalized, 'db1') #ilk dalgacık dönüşümü 
            NorL, NorH= coeffs2
            coeffs2 = pywt.dwt(NorL, 'db1')#2. dalgacık dönüşümü 
            NorLL, NorHL= coeffs2
            coeffs2 = pywt.dwt(NorLL, 'db1')#3. dalgacık dönüşümü 
            NorLLL, NorHLL= coeffs2
            return NorH,NorHL,NorLLL,NorHLL
            # NorH,NorHL,NorLLL,NorHLL=ValidationDatagetitem(validationdata)
        elif self.secim==1:
            noisyalldata = np.load(self.testdata)[idx,:]
            y = noisyalldata
            noisynormalized = (y-min(y))/(max(y)-min(y)+0.0001)
            coeffs2 = pywt.dwt(noisynormalized, 'db1') #ilk dalgacık dönüşümü 
            NL, NH= coeffs2
            coeffs2 = pywt.dwt(NL, 'db1')#2. dalgacık dönüşümü 
            NLL, NHL= coeffs2
            coeffs2 = pywt.dwt(NLL, 'db1')#3. dalgacık dönüşümü 
            NLLL, NHLL= coeffs2
            return NH,NHL,NLLL,NHLL
            # NH,NHL,NLLL,NHLL=TestDatagetitem(noisyalldata)
            
            
            
       
        # print("clearalldata . shape",clearalldata.shape)
        # y_train=self.labels[idx]
        # noisyalldata = np.load(self.testdata)[idx,:]
        # print("noisyalldata . shape",noisyalldata.shape)
        # validationdata=np.load(self.validationdata)[idx,:]
        # print("validationdata . shape",validationdata.shape)
        # print(clearalldata.shape)
        # print(noisyalldata.shape)
        # print(validationdata.shape)
        
        # secim= self. secim(int(input("train / test val için seçimi giriniz")))
        #önce datayı normalize ediyoruz
        #Normalize için 1d için tekrar güncellyim
        #min max aralığına getireceğim
        #verisetinin 
        
        # x =clearalldata
        # clearnormalized = (x-min(x))/(max(x)-min(x)+0.0001)
        
        # y = noisyalldata
        # noisynormalized = (y-min(y))/(max(y)-min(y)+0.0001)
        
        # z=validationdata
        # valnormalized = (z-min(z))/(max(z)-min(z)+0.0001)
        # clearnormalized = preprocessing.normalize(clearalldata)
        # print("Normalized Data = ", clearnormalized)
        # noisynormalized = preprocessing.normalize(noisyalldata)
        # print("Normalized Data = ", noisynormalized)
        
        #dalgacık dönüşümü ıuyguluyoruz her 2 sinyalede
        # coeffs2 = pywt.dwt(clearnormalized, 'db1') #ilk dalgacık dönüşümü 
        # CL, CH= coeffs2
        # coeffs2 = pywt.dwt(CL, 'db1')#2. dalgacık dönüşümü 
        # CLL, CHL= coeffs2
        # coeffs2 = pywt.dwt(CLL, 'db1')#3. dalgacık dönüşümü 
        # CLLL, CHLL= coeffs2
        
        # coeffs2 = pywt.dwt(noisynormalized, 'db1') #ilk dalgacık dönüşümü 
        # NL, NH= coeffs2
        # coeffs2 = pywt.dwt(NL, 'db1')#2. dalgacık dönüşümü 
        # NLL, NHL= coeffs2
        # coeffs2 = pywt.dwt(NLL, 'db1')#3. dalgacık dönüşümü 
        # NLLL, NHLL= coeffs2
        
        # coeffs2 = pywt.dwt(valnormalized, 'db1') #ilk dalgacık dönüşümü 
        # NorL, NorH= coeffs2
        # coeffs2 = pywt.dwt(NL, 'db1')#2. dalgacık dönüşümü 
        # NorLL, NorHL= coeffs2
        # coeffs2 = pywt.dwt(NLL, 'db1')#3. dalgacık dönüşümü 
        # NorLLL, NorHLL= coeffs2
        
        # CH = (torch.from_numpy(CH)[None,:]).float()  #.type(torch.DoubleTensor))
        # # print("Ch shape",CH.shape)
        # CHL = torch.from_numpy(CHL)[None,:].float() #.type(torch.DoubleTensor)
        # # print("Chl shape",CHL.shape)
        # CLLL = torch.from_numpy(CLLL)[None,:].float()#.type(torch.DoubleTensor)
        # # print("Clll shape",CLLL.shape)
        # CHLL = torch.from_numpy(CHLL)[None,:].float()#.type(torch.DoubleTensor)
        # # print("Chll shape",CHLL.shape)
        
        # NH = torch.from_numpy(NH)[None,:].float()#.type(torch.DoubleTensor)
        # # print("Nh shape",NH.shape)
        # NHL = torch.from_numpy(NHL)[None,:].float()#.type(torch.DoubleTensor)
        # # print("nhl shape",NHL.shape)
        # NLLL = torch.from_numpy(NLLL)[None,:].float()#.type(torch.DoubleTensor)
        # # print("nlll shape",NLLL.shape)
        # NHLL = torch.from_numpy(NHLL)[None,:].float()#.type(torch.DoubleTensor)
        # # print("nhll shape",NHLL.shape)
        
        # NorH = torch.from_numpy(NorH)[None,:].float()#.type(torch.DoubleTensor)
        # # print("Nh shape",NH.shape)
        # NorHL = torch.from_numpy(NorHL)[None,:].float()#.type(torch.DoubleTensor)
        # # print("nhl shape",NHL.shape)
        # NorLLL = torch.from_numpy(NorLLL)[None,:].float()#.type(torch.DoubleTensor)
        # # print("nlll shape",NLLL.shape)
        # NorHLL = torch.from_numpy(NorHLL)[None,:].float()#.type(torch.DoubleTensor
        
        
        #ch, tensore dönüştür   transform ile 
        
        

     
        # return CH,CHL,CLLL,CHLL,NH,NHL,NLLL,NHLL,NorH,NorHL,NorLLL,NorHLL

EEGClasificationDataLoader(secim=0)