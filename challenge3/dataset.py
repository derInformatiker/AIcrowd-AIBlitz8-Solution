import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class ImgDataset(Dataset):
    def __init__(self,df,mode,transforms = None):
        self.imageID = df['ImageID']
        self.labels = df['label']
        self.transforms = transforms
        self.mode = mode
        
    def __getitem__(self,x):
        path = self.imageID.iloc[x]
        label = np.array([i for i in str(self.labels.iloc[x])]).astype(int)
        label = np.concatenate((label,np.zeros(23-len(label))+10))
        label = [np.eye(11)[int(i)] for i in label]
        if self.mode == 'train':
            i = cv2.imread(f'data/'+str(path)+'.jpg')[64+32:128+32,64+20:192-20]
        else:
            
            i = cv2.imread(f'data/{self.mode}/'+str(path)+'.jpg')[64+32:128+32,64+20:192-20]
        
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        if self.transforms:
            i = self.transforms(image = i)['image']
            
        i = torch.tensor(i) / 255.0
        i = i.permute(2,0,1)
        if self.mode != 'test':
            return i, torch.Tensor(label), torch.Tensor([float(self.labels.iloc[x])])
        else:
            return i
    
    def __len__(self):
        return len(self.imageID)
    
def getTrainDs(train_tr = None):
    train_df = pd.read_csv('data/trainval.csv')
    return ImgDataset(train_df,'train',train_tr)

def getValDs(val_tr):
    val_df = pd.read_csv('data/val.csv')
    return ImgDataset(val_df,'val',val_tr)

def getTestDs(test_tr):
    val_df = pd.read_csv('data/sample_submission.csv')
    return ImgDataset(val_df,'test',test_tr)

def writeSub(p):
    test_df = pd.read_csv('data/sample_submission.csv')
    