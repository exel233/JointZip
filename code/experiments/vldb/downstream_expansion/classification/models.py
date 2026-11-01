"""PyTorch architecture port of hfawaz's reference Wang2017 ResNet-TSC.

Reference commit e0233efd886df8c6ca18e6f1b545d15aaf423627/classifiers/resnet.py.
Uses the same three 64/128/128 residual blocks and 8/5/3 kernels, with Conv1D
SAME padding, BN, ReLU, GAP and linear logits. It is not the TensorFlow runtime;
training budget, internal validation/refit, and BN runtime arithmetic differ.
"""
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        layers=[]
        for index, kernel in enumerate((8,5,3)):
            layers.extend([nn.Conv1d(incoming if index==0 else outgoing,outgoing,kernel,padding='same'),
                           nn.BatchNorm1d(outgoing,eps=.001,momentum=.01)])
            if index!=2:layers.append(nn.ReLU())
        self.main=nn.Sequential(*layers)
        self.shortcut=nn.Sequential(nn.Conv1d(incoming,outgoing,1),nn.BatchNorm1d(outgoing,eps=.001,momentum=.01)) if incoming!=outgoing else nn.BatchNorm1d(outgoing,eps=.001,momentum=.01)
    def forward(self,x):return torch.relu(self.main(x)+self.shortcut(x))

class ResNetTSC(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.blocks=nn.Sequential(Block(1,64),Block(64,128),Block(128,128))
        self.classifier=nn.Linear(128,classes)
        for m in self.modules():
            if isinstance(m,(nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:nn.init.zeros_(m.bias)
    def forward(self,x):return self.classifier(self.blocks(x).mean(-1))
