# -*- coding: utf-8 -*-


import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import sys, argparse, os
from utils import *

from aeon.datasets import load_classification


class loadorean(Dataset):
    def __init__(self, args, split='train', seed=0):
        super().__init__()
        self.args = args
        self.split = split
        
        if args.dataset == 'JapaneseVowels':
            self.seq_len = 29
        elif args.dataset == 'SpokenArabicDigits':
            self.seq_len = 93
        elif args.dataset == 'CharacterTrajectories':
            self.seq_len = 182
        elif args.dataset == 'InsectWingbeat':
            self.seq_len = 78
        if split in ['train']:
            
            
            if args.dataset == 'InsectWingbeat':
                Xtr, ytr, meta =load_classification(name='InsectWingbeat', split='train',extract_path='../timeclass/dataset/')
            else:
                Xtr, ytr, meta = load_classification(name=args.dataset,split='train')
            # print(Xtr.shape)
            word_to_idx = {}
            for i in range(len(meta['class_values'])):
                word_to_idx[meta['class_values'][i]]=i
                
                
            ytr = [word_to_idx[i] for i in ytr]
            self.label =  F.one_hot(torch.tensor(ytr)).float()    
            self.FeatList = Xtr
            
            
            
        elif split == 'test': 
            if args.dataset == 'InsectWingbeat':
                Xte, yte, meta =load_classification(name='InsectWingbeat', split='test',extract_path='../timeclass/dataset/')
            else:
                Xte, yte, meta = load_classification(name=args.dataset,split='test')
            word_to_idx = {}
            for i in range(len(meta['class_values'])):
                word_to_idx[meta['class_values'][i]]=i
                
            # Xte =torch.from_numpy(Xte).permute(0,2,1).float()
            yte = [word_to_idx[i] for i in yte]
            self.label = F.one_hot(torch.tensor(yte)).float()
            self.FeatList = Xte
            
        
        self.feat_in = self.FeatList[0].shape[0]        
        self.max_len = self.seq_len
        self.num_class =  self.label.shape[-1]
    def __getitem__(self, idx):
        # print(torch.from_numpy(self.FeatList[idx]).shape)
        # print(torch.from_numpy(self.FeatList[idx]).squeeze(0).shape)
        feats = torch.from_numpy(self.FeatList[idx]).permute(1,0).float() #L*d
        
        min_len =self.seq_len
        
        feats = F.pad(feats, pad=(0, 0, min_len-feats.shape[0], 0))
        
        
        label = self.label[idx].float()
         
        return feats, label
         
    def __len__(self):
         return len(self.label)
    
    def proterty(self):
        return self.max_len,self.num_class,self.feat_in
