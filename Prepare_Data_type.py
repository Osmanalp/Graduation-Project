import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy.io
from sklearn.model_selection import train_test_split

calldata=pd.read_csv("data/ndata/EEG_all_epochs.csv")
noalldata=pd.read_csv("data/ndata/EOG_all_epochs.csv")
nalldata=pd.read_csv("data/ndata/EMG_all_epochs.csv")
NumberOfSignals=4514

traindata = []
testdata = []
validationdata = []
trainvaldata=[]
etiketler= ["liste"]
newdata = pd.DataFrame()
for i in range(NumberOfSignals):
    ilk=np.random.randint(2)
    if ilk==0:
        temiz=calldata.iloc[i,:]
        x=temiz
        x=pd.DataFrame(x)
        xT=x.T
        if i==0:
            newdata=xT
            etiketler[0]=0
        else:
            newdata=newdata.append(xT)
            etiketler.append(0)
        
        
        
    else:
        
        s=np.random.randint(2)
        if s==0:
            a=np.random.randint(3400)
            temiz=calldata.iloc[i,:]
            gureye=noalldata.iloc[a,:]
            x =temiz + gureye

            x = pd.DataFrame(x)
            #        xT=x.resahape(1,512)
            xT=x.T
            if i==0:
                newdata=xT
                etiketler[0]=1
            else:
                newdata=newdata.append(xT)
                etiketler.append(1)
            
        elif s==1 :
            b=np.random.randint(5598)
            temiz=calldata.iloc[i,:]
            gurmus=nalldata.iloc[b,:]
            x =temiz + gurmus

            x = pd.DataFrame(x)
            #        xT=x.resahape(1,512)
            xT=x.T
            if i==0:
                newdata=xT
                etiketler[0]=1
            else:
                 newdata=newdata.append(xT)
                 etiketler.append(1)
        # else :
        #     a=np.random.randint(3400)
        #     b=np.random.randint(5598)
        #     gureye=noalldata.iloc[a,:]
        #     gurmus=nalldata.iloc[b,:]
        #     temiz=calldata.iloc[i,:]
        #     newgureye=gureye*np.random.rand(0,1)
        #     newgurmus=gurmus*np.random.rand(0,1)
        #     x=temiz+newgureye+newgurmus
        #     x=pd.DataFrame(x)
        #     xT=x.T
        #     if i==0:
        #         newdata=xT
        #         etiketler=1
        #     else:
        #         newdata=newdata.append(xT)
        #         etiketler.append(1)
           
# print(type(newdata)) 
      
X_train, X_test, y_train, y_test = train_test_split(newdata,etiketler,test_size=0.1,random_state=2)   
print(type(X_train))
# newdata=np.array(newdata) 
# newdata,testdata= train_test_split(newdata,test_size=0.1,random_state=2)  
# print(len(newdata))   
# print(len(testdata)) 
# X_train=newdata
# X_train, X_val=train_test_split(X_train,test_size=0.1,random_state=2)    
# print(len())            
# X_train=np.array(newdata)
# # print("calldata.shape" , calldata.shape)
# testdata=np.array(calldata)
# # print("testdata shape" , testdata.shape)

# # traindata, validationdata, testdata = np.split(newdata.sample(frac=1), [int(.6*len(newdata)), int(.8*len(newdata))])
# X_train,X_val =train_test_split(X_train,test_size=0.1,random_state=2)
# # print("X_train.shape:", X_train.shape)  
# # print("X_val.shape:", X_val.shape) 

# np.save("X_train.npy",X_train)
# np.save("X_test.npy",X_test)
# np.save("Y_train.npy",y_train)
# np.save("Y_test.npy",y_test)



# traindata,validationdata=train_test_split(traindata, validationdata, validation_size=0.2)
                                                   
# traindata.to_csv('data/ndata/traindata.csv', index=False) 
# validationdata.to.csv('data/ndata/validationdata.csv', index=False)
# testdata.to_csv('data/ndata/testdata.csv', index=False)       

       
        
    
    
 
   