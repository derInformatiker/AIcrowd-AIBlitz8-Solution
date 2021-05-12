import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm
from detecto import core, utils, visualize
import numpy as np

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
        
        i = np.load(f'data/preprocessed/{self.mode}/'+str(path)+'.npy')
        
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

def preprocess():
    model = core.Model.load('best.pth',['car'])
    
    def process_image(image):
        labels, boxes, scores = model.predict(image)
        y,x,h,w = boxes[0]
        image = image[int(x):int(w),int(y):int(h),:][:160,:160]

        hh,ww = 160,160
        h, w = image.shape[:2]

        yoff = round((hh-h)/2)
        xoff = round((ww-w)/2)

        result = np.zeros((hh,ww,3)).astype('int')
        result[yoff:yoff+h, xoff:xoff+w] = image
        result = result.astype('uint8')
        return result
    
    for i in tqdm(range(40000)):
        image = utils.read_image(f'data/train/{i}.jpg')
        result = process_image(image)
        np.save(f'data/preprocessed/train/{i}.npy',result)
    
    for i in tqdm(range(4000)):
        image = utils.read_image(f'data/val/{i}.jpg')
        result = process_image(image)
        np.save(f'data/preprocessed/val/{i}.npy',result)
        
    for i in tqdm(range(10000)):
        image = utils.read_image(f'data/test/{i}.jpg')
        result = process_image(image)
        np.save(f'data/preprocessed/test/{i}.npy',result)