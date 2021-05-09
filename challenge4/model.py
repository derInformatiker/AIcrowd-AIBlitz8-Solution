import os
import dataset
import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet


class Classifier(pl.LightningModule):

    def __init__(self,args):
        super().__init__()
        self.args = args
        
        #self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False,num_classes = 2)
        self.resnet = EfficientNet.from_pretrained('efficientnet-b4',num_classes = 4)
        #self.resnet._avg_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = self.resnet(x)
        return x
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        p = self(x)
        loss = F.cross_entropy(p, y.long())
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        p = self(x)
        loss = F.cross_entropy(p, y.long())
        f1 = f1_loss(y,p.argmax(1))
        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        self.log('val_f1',f1)
        return loss
    
    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        
        x, y = batch
        p = self(x)
        
        loss = F.cross_entropy(p, y.long())
        mse = F.mse_loss(p.argmax(1),y.float())
        # Logging to TensorBoard by default
        self.log('test_loss', loss)
        self.log('test_mse',mse)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
    
    def train_dataloader(self):
        train_ds = dataset.getTrainDs(self.args['train_tr'])
        loader= DataLoader(train_ds,batch_size = self.args['batch_size'],num_workers = 8,shuffle=True)
        return loader
    
    def val_dataloader(self):
        val_ds = dataset.getValDs(self.args['val_tr'])
        loader= DataLoader(val_ds,batch_size = self.args['batch_size'],num_workers = 8)
        return loader
    
    def test_dataloader(self):
        val_ds = dataset.getValDs(self.args['val_tr'])
        loader= DataLoader(val_ds,batch_size = self.args['batch_size'],num_workers = 8)
        return loader
    
    def predict_dataloader(self):
        val_ds = dataset.getTestDs(self.args['val_tr'])
        loader= DataLoader(val_ds,batch_size = self.args['batch_size'],num_workers = 8)
        return loader
    
    
def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:

    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)


    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def writeSub(p):
    test_df = pd.read_csv('data/sample_submission.csv')
    p = p+1
    output_list = p.int().tolist()
    test_df['label'] = output_list
    test_df.to_csv('submission.csv',index = False)