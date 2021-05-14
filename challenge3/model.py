import os
import dataset
import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet

# CODE FROM https://github.com/meijieru/crnn.pytorch
# LICENSE: MIT-LICENSE

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
    
class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        cnn.add_module('adaptive_pooling0',
                        nn.AdaptiveAvgPool2d((1,32)))

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        return output
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
        self.model = CRNN(128,3,11,256)

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