import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

class ImgDataset(Dataset):
    def __init__(self,df,mode,transforms = None):
        self.imageID = df['ImageID']
        self.labels = df['label']
        self.labelmap = {'right':0, 'left':1, 'front':2, 'back':3}
        self.transforms = transforms
        self.mode = mode
        
    def __getitem__(self,x):
        path = self.imageID.iloc[x]
        label = float(self.labelmap[self.labels.iloc[x]])
        
        i = cv2.imread(f'data/{self.mode}/'+str(path)+'.jpg')
        
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        if self.transforms:
            i = self.transforms(image = i)['image']
        i = torch.tensor(i) / 255.0
        i = i.permute(2,0,1)
        if self.mode != 'test':
            return i, label
        else:
            return i
    
    def __len__(self):
        return len(self.imageID)
    
def getTrainDs(train_tr = None):
    train_df = pd.read_csv('data/train.csv')
    return ImgDataset(train_df,'train',train_tr)

def getValDs(val_tr):
    val_df = pd.read_csv('data/val.csv')
    return ImgDataset(val_df,'val',val_tr)

def getTestDs(test_tr):
    val_df = pd.read_csv('data/sample_submission.csv')
    return ImgDataset(val_df,'test',test_tr)

def writeSub(p):
    test_df = pd.read_csv('data/sample_submission.csv')
    