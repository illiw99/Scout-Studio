"""Optional reference-only temporal heads; runtime inference uses numeric ONNX."""
from __future__ import annotations
import numpy as np


def build_head(width_in, tokens, *, architecture='transformer', width=32, layers=2,
               mean=None, scale=None):
    import torch
    from torch import nn
    class AttentionBlock(nn.Module):
        def __init__(self):
            super().__init__();self.norm=nn.LayerNorm(width);self.qkv=nn.Linear(width,width*3)
            self.project=nn.Linear(width,width);self.norm2=nn.LayerNorm(width)
            self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Dropout(.1),nn.Linear(width*2,width))
        def forward(self,x):
            q,k,v=self.qkv(self.norm(x)).chunk(3,dim=-1)
            attention=(q@k.transpose(-1,-2)/(width**.5)).softmax(dim=-1)
            x=x+self.project(attention@v)
            return x+self.ff(self.norm2(x))
    class Head(nn.Module):
        def __init__(self):
            super().__init__();self.project=nn.Linear(width_in,width)
            self.register_buffer('mean',torch.tensor(np.zeros(width_in) if mean is None else mean,dtype=torch.float32))
            self.register_buffer('scale',torch.tensor(np.ones(width_in) if scale is None else scale,dtype=torch.float32))
            self.position=nn.Parameter(torch.zeros(1,tokens,width))
            if architecture=='transformer':self.blocks=nn.ModuleList([AttentionBlock() for _ in range(layers)])
            elif architecture=='tcn':self.blocks=nn.ModuleList([nn.Conv1d(width,width,3,padding=2**i,dilation=2**i) for i in range(layers)])
            else:raise ValueError('Unknown temporal architecture')
            self.output=nn.Linear(width,1)
        def forward(self,raw):
            raw=torch.where(torch.isnan(raw),self.mean,raw)
            x=self.project(((raw-self.mean)/self.scale).clamp(-8,8))+self.position
            for block in self.blocks:
                x=block(x) if architecture=='transformer' else x+torch.relu(block(x.transpose(1,2))).transpose(1,2)
            return self.output(x[:,tokens//2]).squeeze(-1)
    return Head()


def normalization(cases):
    X=np.concatenate([c['X'] for c in cases]).astype(np.float32)
    with np.errstate(invalid='ignore'):
        count=np.isfinite(X).sum(axis=0);mean=np.nansum(X,axis=0)/np.maximum(count,1)
        variance=np.nansum((X-mean)**2,axis=0)/np.maximum(count,1)
    return mean.astype(np.float32),np.maximum(np.sqrt(variance),.05).astype(np.float32)
