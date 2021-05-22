import os
import dataset
import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# CODE FROM https://github.com/clovaai/deep-text-recognition-benchmark
"""
Copyright (c) 2019-present NAVER Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        """ Transformation """
        
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=20, I_size=(64, 80), I_r_size=(64, 80), I_channel_num=3)
        

        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(3, 512)
        
        self.FeatureExtraction_output = 512  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
            BidirectionalLSTM(256, 256, 256))
        self.SequenceModeling_output = 256
        

        """ Prediction """
        
        self.Prediction = nn.Linear(self.SequenceModeling_output, 11)

    def forward(self, input, is_train=True):
        """ Transformation stage """
        #input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        
        contextual_feature = self.SequenceModeling(visual_feature)


        """ Prediction stage """
        
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction.permute(1,0,2)

def toNum(t):
    output = []
    t = t.argmax(2)
    for label in t:
        try:
            l = []
            for i in label.tolist():
                if i == 10:
                    break
                l.append(str(i))
            output.append(int(''.join(l)))
        except:
            output.append(0)
    return torch.Tensor(output).unsqueeze(1).cuda()
    
    
class Classifier(pl.LightningModule):

    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model = Model()
        c = torch.load('TPS-ResNet-BiLSTM-CTC.pth')
        c = {k.replace('module.',''):v for k,v in c.items() if 'LocalizationNetwork.conv.0.weight' not in k and 'FeatureExtraction.ConvNet.conv0_1.weight' not in k and
        'Prediction' not in k and 'Transformation.GridGenerator.P_hat' not in k}
        self.model.load_state_dict(c,strict = False)
        
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = self.model(x)
        return x.permute(1,0,2)
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, num = batch
        p = self(x)
        loss = F.binary_cross_entropy_with_logits(p, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, num = batch
        p = self(x)
        loss = F.binary_cross_entropy_with_logits(p, y)
        mse = F.mse_loss(toNum(p),num)
        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        self.log('val_mse',mse)
        return loss
    
    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, num = batch
        p = self(x)
        
        loss = F.binary_cross_entropy_with_logits(p, y)
        mse = F.mse_loss(toNum(p),num)
        # Logging to TensorBoard by default
        self.log('test_loss', loss)
        self.log('test_mse',mse)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
    
    def train_dataloader(self):
        train_ds = dataset.getTrainDs(self.args['train_tr'])
        loader= DataLoader(train_ds,batch_size = self.args['batch_size'],num_workers = 4,shuffle=True)
        return loader
    
    def val_dataloader(self):
        val_ds = dataset.getValDs(self.args['val_tr'])
        loader= DataLoader(val_ds,batch_size = self.args['batch_size'],num_workers = 4)
        return loader
    
    def test_dataloader(self):
        val_ds = dataset.getValDs(self.args['val_tr'])
        loader= DataLoader(val_ds,batch_size = self.args['batch_size'],num_workers = 4)
        return loader
    
    def predict_dataloader(self):
        val_ds = dataset.getTestDs(self.args['val_tr'])
        loader= DataLoader(val_ds,batch_size = self.args['batch_size'],num_workers = 4)
        return loader


def writeSub(p):
    test_df = pd.read_csv('data/sample_submission.csv')
    output_list = p.int().tolist()
    test_df['label'] = output_list
    test_df.to_csv('submission.csv',index = False)