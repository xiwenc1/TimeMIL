# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
#from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_fscore_support,f1_score,accuracy_score,precision_score,recall_score,balanced_accuracy_score
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict

# from models.dropout import LinearScheduler
from aeon.datasets import load_classification

from utils import *

from torch.cuda.amp import GradScaler, autocast
                                        
                                        
import torch.nn.functional as F

from mydataload import loadorean

import random

# from sample_method import rd_torch,dpp

import random
from timm.optim.adamp import AdamP
from lookhead import Lookahead
import warnings



from models.timemil import TimeMIL

# Suppress all warnings
warnings.filterwarnings("ignore")

                                  


def train(trainloader, milnet, criterion, optimizer, epoch,args):
    milnet.train()
    total_loss = 0


    for batch_id, (feats, label) in enumerate(trainloader):
    #     bag_feats = feats.cuda()
    #     bag_label = label.cuda()

        bag_feats = feats.cuda()
        bag_label = label.cuda()
        
        # print(bag_feats.shape)
        
        # window-based random masking
        if args.dropout_patch>0:
            selecy_window_indx = random.sample(range(10),int(args.dropout_patch*10))
            inteval = int(len(bag_feats)//10)
            
            for idx in selecy_window_indx:
                bag_feats[:,idx*inteval:idx*inteval+inteval,:] = torch.randn(1).cuda()
        
   

        optimizer.zero_grad()
   
        if epoch<args.epoch_des:
            bag_prediction  = milnet(bag_feats,warmup = True)
        else:
            bag_prediction  = milnet(bag_feats,warmup = False)
       
        bag_loss = criterion(bag_prediction, bag_label)
       
        if True:
            loss = bag_loss 
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f  total loss: %.4f' % \
                            (batch_id, len(trainloader), bag_loss.item(),loss.item()))
        
        
     
        loss.backward()
        
    
        # avoid the overfitting by using gradient clip
        torch.nn.utils.clip_grad_norm_(milnet.parameters(), 2.0)
        optimizer.step()

        # total_loss = total_loss + loss.item()
        total_loss = total_loss + bag_loss
      

    return total_loss / len(trainloader)



def test(testloader, milnet, criterion, args):
    milnet.eval()
    # csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    
    with torch.no_grad():
        
        for batch_id, (feats, label) in enumerate(testloader):
            
            
            bag_feats = feats.cuda()
            bag_label = label.cuda()
         
            bag_prediction = milnet(bag_feats)  #b*class
           

            bag_loss = criterion(bag_prediction,bag_label)
            
            
            
            loss = bag_loss
            total_loss = total_loss + loss.item()

            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (batch_id, len(testloader), loss.item()))
            
            # test_labels.extend([label.squeeze().cpu().numpy()])
            test_labels.extend([label.cpu().numpy()])
            test_predictions.extend([torch.sigmoid(bag_prediction).cpu().numpy()])
    
    test_labels = np.vstack(test_labels)
    test_predictions = np.vstack(test_predictions)

    test_predictions_prob = np.exp(test_predictions)/np.sum(np.exp(test_predictions),axis=1,keepdims=True)
    
    
    test_predictions = np.argmax(test_predictions,axis=1)
    test_labels = np.argmax(test_labels,axis=1)
    avg_score = accuracy_score(test_labels,test_predictions)
    balanced_avg_score = balanced_accuracy_score(test_labels,test_predictions)
    f1_marco = f1_score(test_labels,test_predictions,average='macro')
    f1_micro = f1_score(test_labels,test_predictions,average='micro')
    
    p_marco = precision_score(test_labels,test_predictions,average='macro')
    p_micro = precision_score(test_labels,test_predictions,average='micro')
    
    r_marco = recall_score(test_labels,test_predictions,average='macro')
    r_micro = recall_score(test_labels,test_predictions,average='micro')
    
    
    r_marco = recall_score(test_labels,test_predictions,average='macro')
    r_micro = recall_score(test_labels,test_predictions,average='micro')
    
    if args.num_classes ==2: 
        # print(test_labels.shape)
        
        roc_auc_ovo_marco=   roc_auc_score(test_labels,test_predictions_prob[:,1],average='macro')
        roc_auc_ovo_micro=  0.# roc_auc_score(test_labels,test_predictions_prob,average='micro',multi_class='ovo')
    
        roc_auc_ovr_marco=   roc_auc_score(test_labels,test_predictions_prob[:,1],average='macro')
        roc_auc_ovr_micro=  0.# roc_auc_score(test_labels,test_predictions_prob,average='micro',multi_class='ovr')
    
    
    else:
        roc_auc_ovo_marco=   roc_auc_score(test_labels,test_predictions_prob,average='macro',multi_class='ovo')
        roc_auc_ovo_micro=  0.# roc_auc_score(test_labels,test_predictions_prob,average='micro',multi_class='ovo')

        roc_auc_ovr_marco=   roc_auc_score(test_labels,test_predictions_prob,average='macro',multi_class='ovr')
        roc_auc_ovr_micro=  0.# 
    

    
    results = [avg_score,balanced_avg_score,f1_marco,f1_micro, p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro]
    
    
    return total_loss / len(testloader), results



def main():
    parser = argparse.ArgumentParser(description='time classification by TimeMIL')
    parser.add_argument('--dataset', default="PenDigits", type=str, help='dataset ')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512] resnet-50 1024')
    parser.add_argument('--lr', default=5e-3, type=float, help='1e-3 Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=300, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay 1e-4]')
    parser.add_argument('--dropout_patch', default=0.5, type=float, help='Patch dropout rate [0] 0.5')
    parser.add_argument('--dropout_node', default=0.2, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--seed', default='0', type=int, help='random seed')
   
    parser.add_argument('--optimizer', default='adamw', type=str, help='adamw sgd')
    
    parser.add_argument('--save_dir', default='./savemodel/', type=str, help='the directory used to save all the output')
    parser.add_argument('--epoch_des', default=10, type=int, help='turn on warmup')
   
    parser.add_argument('--embed', default=128, type=int, help='Number of embedding')
    
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
    
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    args.save_dir = args.save_dir+'InceptBackbone'
    
    maybe_mkdir_p(join(args.save_dir, f'{args.dataset}'))
    args.save_dir = make_dirs(join(args.save_dir, f'{args.dataset}'))
    maybe_mkdir_p(args.save_dir)
    


    # <------------- set up logging ------------->
    logging_path = os.path.join(args.save_dir, 'Train_log.log')
    logger = get_logger(logging_path)

    # <------------- save hyperparams ------------->
    option = vars(args)
    file_name = os.path.join(args.save_dir, 'option.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')


    # criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.0)#0.01
    criterion = nn.BCEWithLogitsLoss() # one-vs-rest binary MIL
    # scaler = GradScaler()
    
    ###################################
    if args.dataset in ['JapaneseVowels','SpokenArabicDigits','CharacterTrajectories','InsectWingbeat']:
       
        trainset = loadorean(args, split='train')
        testset = loadorean(args, split='test')
        
        seq_len,num_classes,L_in=trainset.max_len,trainset.num_class,trainset.feat_in
        
        
        
        
        print(f'max lenght {seq_len}')
        args.feats_size = L_in
        args.num_classes =  num_classes
        print(f'num class:{args.num_classes}' )
        
    else:
        Xtr, ytr, meta = load_classification(name=args.dataset,split='train')

        print(Xtr.shape)
        word_to_idx = {}
        for i in range(len(meta['class_values'])):
            word_to_idx[meta['class_values'][i]]=i

        Xtr =torch.from_numpy(Xtr).permute(0,2,1).float()
        ytr = [word_to_idx[i] for i in ytr]
        ytr =  F.one_hot(torch.tensor(ytr)).float()
        print(ytr.shape)

        trainset = TensorDataset(Xtr,ytr)

        Xte, yte, _ = load_classification(name=args.dataset,split='test')

        Xte =torch.from_numpy(Xte).permute(0,2,1).float()
        yte = [word_to_idx[i] for i in yte]
        yte = F.one_hot(torch.tensor(yte)).float()


        testset = TensorDataset(Xte,yte)




        args.feats_size = Xte.shape[-1]
        L_in = Xte.shape[-1]
        # print(L_in)
        num_classes = yte.shape[-1]
        args.num_classes =  yte.shape[-1]
        print(f'num class:{args.num_classes}' )
        # args.num_classes = num_classes

        seq_len=  max(21, Xte.shape[1])
    
    
    
    
    # <------------- define MIL network ------------->
   
    
    milnet = TimeMIL(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len).cuda()
    
    if  args.optimizer == 'adamw':
    # optimizer = torch.optim.AdamW(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(milnet.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer)    
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(milnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        # optimizer =Lookahead(optimizer) 
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer) 
    
    elif args.optimizer == 'adamp':
        optimizer = AdamP(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer) 
    
    

    trainloader = DataLoader(trainset, args.batchsize, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    # if args.batchsize==1:
    #     testloader = DataLoader(testset, args.batchsize, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    # else:
    testloader = DataLoader(testset, 128, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    

    best_score = 0
    save_path = join(args.save_dir, 'weights')
    os.makedirs(save_path, exist_ok=True)
    
    os.makedirs(join(args.save_dir,'lesion'), exist_ok=True)
    results_best = None
    for epoch in range(1, args.num_epochs + 1):

        train_loss_bag = train(trainloader, milnet, criterion, optimizer, epoch,args) # iterate all bags
        
      
        test_loss_bag,results= test(testloader, milnet, criterion, args)
      
        [avg_score,balanced_avg_score,f1_marco,f1_micro,p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro] = results
    
        

        
        logger.info('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, accuracy: %.4f, bal. average score: %.4f, f1 marco: %.4f   f1 mirco: %.4f  p marco: %.4f   p mirco: %.4f r marco: %.4f   r mirco: %.4f  roc_auc ovo marco: %.4f   roc_auc ovo mirco: %.4f  roc_auc ovr marco: %.4f   roc_auc ovr mirco: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score,balanced_avg_score,f1_marco,f1_micro, p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro )) 
        
        
        # scheduler.step()
        # current_score = (sum(aucs) + avg_score)/3
        current_score = avg_score
        if current_score >= best_score:
            
            results_best = results
            
            best_score = current_score
            print(current_score)
            save_name = os.path.join(save_path, 'best_model.pth')
            torch.save(milnet.state_dict(), save_name)
            #torch.save(milnet, save_name)
            logger.info('Best model saved at: ' + save_name)
            # logger.info('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
    
    
    [avg_score,balanced_avg_score,f1_marco,f1_micro,p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro] = results_best
    logger.info('\r Best  Results: accuracy: %.4f, bal. average score: %.4f, f1 marco: %.4f   f1 mirco: %.4f  p marco: %.4f   p mirco: %.4f r marco: %.4f   r mirco: %.4f  roc_auc ovo marco: %.4f   roc_auc ovo mirco: %.4f  roc_auc ovr marco: %.4f   roc_auc ovr mirco: %.4f' % 
                  ( avg_score,balanced_avg_score,f1_marco,f1_micro, p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro )) 
    
        # if args.weight_div>0:
        #     if epoch%10==0:
        #         print('--------------------Clustering--------------------\n')
        #         cluster_idx_dict = pre_cluter(trainloader, milnet, criterion, optimizer, args,init= False)
        #         print('--------------------Clustering finished--------------------\n')

if __name__ == '__main__':
    main()
