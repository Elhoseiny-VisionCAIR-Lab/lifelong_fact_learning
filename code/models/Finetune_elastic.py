#!/usr/bin/env python3

"""
Created on Wed Sep 20 19:07:53 2017
@author: raljundi& fbabilon

Train or Finetune a Fact model using Elastic SGD 'paper_ref'.
 - To Train from stratch use : previous_task_model_path = ''  
 - To finetune from previous task use : previous_task_model_path= "rel_link_to_your_model" 

"""

#adding optimizer packages
import sys

# Implement Sherlock Paper in Pytorch
##################################################################################################################
import pandas as pd
import pdb
import os
import collections
import torch
import numpy as np
import torch.nn.parallel
from torchvision import transforms, models
from utils import train_eval_utils
from utils import sherlock_model_utils
from utils import dataset_utils
from utils import elastic_caller
from utils.Elastic_utils import Elastic_Training
def fill_default_vals(root,previous_task_model_path,train_data_path,test_data_path,exp_dir,reg_lambda,epochs,lr,test_iter,batch):
    func_params=[]
    
    print('YOUR PARAMETERS')
  
    if root is None:
        root= '/users/visics/fbabilon/Sherlock/export_Sherlock/'
    print('the root directory is ',root) 
        
    if reg_lambda is None:
        reg_lambda=1
    print('the regularizer multipler value is ',reg_lambda)
        
    if exp_dir is None:    
        ##############################WARNING CHANGE TEST HERE#####################################
        exp_dir=root+'pytorch_models/B2_elastic'+'/reg_'+str(reg_lambda)+'/lr_06/'
    print('the exporting directory is ',exp_dir)
        
    if previous_task_model_path is None: 
        #in case of Task 2, set this path to your model trained on Task1 
        previous_task_model_path= root+'pytorch_models'+'/B1_elastic/'  +'model_best.pth.tar'
        #previous_task_model_path=previous_task_model_path+'model_best.pth.tar'
        
        #in case of Task 1, set this path to empty
    print('the previous task model is ',previous_task_model_path)
        
    if train_data_path is None:
        train_data_path=root + 'data_splits/B2_train.csv'
    print('the training data path is ',train_data_path)
        
    if test_data_path is None:
        test_data_path=root + 'data_splits/B2_test.csv'
    print('the test data path is ',test_data_path)
        
    if epochs is None:    
        epochs = 1750  #epoch_no = max_iter(700000)*batch_size(35)/total_number_of_samples_training(14025)
        print('The training will run for this number of epochs:',epochs)
        
    if test_iter is None:
        test_iter = 5#12
    print('The training will check the performance on eval set every ',test_iter)
        
    if batch is None:
        batch=35
    print('Training will be  done with a batch size of ', batch)
        
    if lr is None:
        lr=5e-5
    
    print('The base learning rate is ',lr)
    
    
    
        
    return (root,previous_task_model_path,train_data_path,test_data_path,exp_dir,reg_lambda,epochs,lr,test_iter,batch)

def finetune_elastic(root=None,previous_task_model_path=None,train_data_path=None,test_data_path=None,exp_dir=None,reg_lambda=None,epochs=None,lr=None,test_iter=None,batch=None,use_multiple_gpu=None):
    
    'DEFINE TRAINING PARAMETERS'
    root,previous_task_model_path,train_data_path,test_data_path,exp_dir,reg_lambda,epochs,lr,test_iter,batch=fill_default_vals(root,previous_task_model_path,train_data_path,test_data_path,exp_dir,reg_lambda,epochs,lr,test_iter,batch)
    
    #make your save directory
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
 
    #define LOSS
    criterion = sherlock_model_utils.Fact_Euclidian_Loss()
    print ('Criterion:', type(criterion))

    # in case you have a checkpoint saved in the resume path, the training will start from there
    resume_path = exp_dir+ '/checkpoint.pth.tar'


    'LOAD YOUR DATASET INFORMATION'
    r"""

    df_train and df_test are dataframe holding:
        - image_links: rel_path to each image
        - NLP_links:   rel_path to each NLP representation of each image
        - SPO: fact representation S:subject, P:Predicate, O:Object
        - id : fact label. (each fact has its own Unique label)
        - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

    """
    df_train = pd.read_csv(train_data_path)
    df_test= pd.read_csv(test_data_path)
    
    ##################################################################################################################

    'MAKE YOUR  DATASET'
    # your dataloader will hold
    # images, NLP, ws, wp, wo, labels = data 

    train_dt = dataset_utils.Cst_Dataset(df_train['image_links'], df_train['NLP_links'], df_train['id'], df_train['w_s'] , df_train['w_p'], df_train['w_o'], 
                          root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader,  transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.RandomSizedCrop(224),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std
                                                                        ]))


    test_dt = dataset_utils.Cst_Dataset(df_test['image_links'], df_test['NLP_links'], df_test['id'], df_test['w_s'] , df_test['w_p'], df_test['w_o'] ,
                         root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader, transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                        ]))




    del  df_train, df_test

    # Make your dataset accessible in batches
    dset_loaders = {'train': torch.utils.data.DataLoader(train_dt, batch_size=batch, shuffle=True, num_workers=4),
                    'val':torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=True, num_workers=4)}


    use_gpu = torch.cuda.is_available()  
    print('Training will be done using gpu %s'%use_gpu)
    
    ##################################################################################################################
    
    'LOAD YOUR INITIAL MODEL'
    
    #loading previous model   
    Sherlock_Net=[]
    
    if not os.path.isfile(resume_path):
        checkpoint=[]
        
        #TRAINING ON TASK 1
        if not os.path.isfile(previous_task_model_path):
            #build your Sherlock Net from stratch            
            Sherlock_Net = sherlock_model_utils.build_Sherlock_Net()
            
            #initialize Sherloch Net with VGG16 params
            Sherlock_Net = sherlock_model_utils.initialize_from_VGG(Sherlock_Net, use_gpu)
                 
        
        #FINETUNING
        else:
            #Importing model from previous task.
            print('Loading model from a previous task') 
            Sherlock_Net=torch.load(previous_task_model_path)
            
        #set model on GPU
    
        if use_gpu:
            Sherlock_Net=Sherlock_Net.cuda()
            
    else:
        checkpoint = torch.load(resume_path)
        Sherlock_Net = checkpoint['model']
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume_path, checkpoint['epoch']))

            

    'SET LEARNING RATE AND WEIGHT DECAY'
    params=train_eval_utils.set_param_learning_rate(Sherlock_Net,lr)

    'TRAIN'
    reg_lambda=float(reg_lambda)
    elastic_caller.finetune_elastic(dset_loaders,Sherlock_Net,params,criterion, epochs,exp_dir,checkpoint,lr,test_iter,reg_lambda,use_multiple_gpu)

    
# eccv add
def finetune_elastic_model2(root=None,previous_task_model_path=None,train_data_path=None,test_data_path=None,exp_dir=None,reg_lambda=None,epochs=None,lr=None,test_iter=None,batch=None,use_multiple_gpu=None):
    
    'DEFINE TRAINING PARAMETERS'
    root,previous_task_model_path,train_data_path,test_data_path,exp_dir,reg_lambda,epochs,lr,test_iter,batch=fill_default_vals(root,previous_task_model_path,train_data_path,test_data_path,exp_dir,reg_lambda,epochs,lr,test_iter,batch)
    
    #make your save directory
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
 
    #define LOSS
    criterion = sherlock_model_utils.Fact_Euclidian_Loss()
    print ('Criterion:', type(criterion))

    # in case you have a checkpoint saved in the resume path, the training will start from there
    resume_path = exp_dir+ '/checkpoint.pth.tar'


    'LOAD YOUR DATASET INFORMATION'
    r"""

    df_train and df_test are dataframe holding:
        - image_links: rel_path to each image
        - NLP_links:   rel_path to each NLP representation of each image
        - SPO: fact representation S:subject, P:Predicate, O:Object
        - id : fact label. (each fact has its own Unique label)
        - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

    """
    df_train = pd.read_csv(train_data_path)
    df_test= pd.read_csv(test_data_path)
    
    ##################################################################################################################

    'MAKE YOUR  DATASET'
    # your dataloader will hold
    # images, NLP, ws, wp, wo, labels = data 

    train_dt = dataset_utils.Cst_Dataset(df_train['image_links'], df_train['NLP_links'], df_train['id'], df_train['w_s'] , df_train['w_p'], df_train['w_o'], 
                          root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader,  transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.RandomSizedCrop(224),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std
                                                                        ]))


    test_dt = dataset_utils.Cst_Dataset(df_test['image_links'], df_test['NLP_links'], df_test['id'], df_test['w_s'] , df_test['w_p'], df_test['w_o'] ,
                         root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader, transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                        ]))




    del  df_train, df_test

    # Make your dataset accessible in batches
    dset_loaders = {'train': torch.utils.data.DataLoader(train_dt, batch_size=batch, shuffle=True, num_workers=4),
                    'val':torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=True, num_workers=4)}


    use_gpu = torch.cuda.is_available()  
    print('Training will be done using gpu %s'%use_gpu)
    
    ##################################################################################################################
    
    'LOAD YOUR INITIAL MODEL'
    
    #loading previous model   
    Sherlock_Net=[]
    
    if not os.path.isfile(resume_path):
        checkpoint=[]
        
        #TRAINING ON TASK 1
        if not os.path.isfile(previous_task_model_path):
            #build your Sherlock Net from stratch            
            Sherlock_Net = sherlock_model_utils.build_Sherlock_Net()
            
            #initialize Sherloch Net with VGG16 params
            Sherlock_Net = sherlock_model_utils.initialize_model2_from_VGG(Sherlock_Net, use_gpu)
                 
        
        #FINETUNING
        else:
            #Importing model from previous task.
            print('Loading model from a previous task') 
            Sherlock_Net=torch.load(previous_task_model_path)
            
        #set model on GPU
    
        if use_gpu:
            Sherlock_Net=Sherlock_Net.cuda()
            
    else:
        checkpoint = torch.load(resume_path)
        Sherlock_Net = checkpoint['model']
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume_path, checkpoint['epoch']))

            

    'SET LEARNING RATE AND WEIGHT DECAY'
    params=train_eval_utils.set_param_learning_rate(Sherlock_Net,lr)

    'TRAIN'
    reg_lambda=float(reg_lambda)
    elastic_caller.finetune_elastic(dset_loaders,Sherlock_Net,params,criterion, epochs,exp_dir,checkpoint,lr,test_iter,reg_lambda,use_multiple_gpu)
#end eccv add
#################     
    
def finetune_elastic_model1(root=None,previous_task_model_path=None,train_data_path=None,test_data_path=None,exp_dir=None,reg_lambda=None,epochs=None,lr=None,test_iter=None,batch=None,use_multiple_gpu=None):
    
    'DEFINE TRAINING PARAMETERS'
    root,previous_task_model_path,train_data_path,test_data_path,exp_dir,reg_lambda,epochs,lr,test_iter,batch=fill_default_vals(root,previous_task_model_path,train_data_path,test_data_path,exp_dir,reg_lambda,epochs,lr,test_iter,batch)
    
    #make your save directory
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
 
    #define LOSS
    criterion = sherlock_model_utils.Fact_Euclidian_Loss()
    print ('Criterion:', type(criterion))

    # in case you have a checkpoint saved in the resume path, the training will start from there
    resume_path = exp_dir+ '/checkpoint.pth.tar'


    'LOAD YOUR DATASET INFORMATION'
    r"""

    df_train and df_test are dataframe holding:
        - image_links: rel_path to each image
        - NLP_links:   rel_path to each NLP representation of each image
        - SPO: fact representation S:subject, P:Predicate, O:Object
        - id : fact label. (each fact has its own Unique label)
        - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

    """
    df_train = pd.read_csv(train_data_path)
    df_test= pd.read_csv(test_data_path)
    
    ##################################################################################################################

    'MAKE YOUR  DATASET'
    # your dataloader will hold
    # images, NLP, ws, wp, wo, labels = data 

    train_dt = dataset_utils.Cst_Dataset(df_train['image_links'], df_train['NLP_links'], df_train['id'], df_train['w_s'] , df_train['w_p'], df_train['w_o'], 
                          root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader,  transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.RandomSizedCrop(224),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std
                                                                        ]))


    test_dt = dataset_utils.Cst_Dataset(df_test['image_links'], df_test['NLP_links'], df_test['id'], df_test['w_s'] , df_test['w_p'], df_test['w_o'] ,
                         root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader, transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                        ]))




    del  df_train, df_test

    # Make your dataset accessible in batches
    dset_loaders = {'train': torch.utils.data.DataLoader(train_dt, batch_size=batch, shuffle=True, num_workers=4),
                    'val':torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=True, num_workers=4)}


    use_gpu = torch.cuda.is_available()  
    print('Training will be done using gpu %s'%use_gpu)
    
    ##################################################################################################################
    
    'LOAD YOUR INITIAL MODEL'
    
    #loading previous model   
    Sherlock_Net=[]
    
    if not os.path.isfile(resume_path):
        checkpoint=[]
        
        #TRAINING ON TASK 1
        if not os.path.isfile(previous_task_model_path):
            #build your Sherlock Net from stratch            
            Sherlock_Net = sherlock_model_utils.build_model1_Sherlock_Net()
            
            #initialize Sherloch Net with VGG16 params
            Sherlock_Net = sherlock_model_utils.initialize_model1_from_VGG(Sherlock_Net, use_gpu)
                 
        
        #FINETUNING
        else:
            #Importing model from previous task.
            print('Loading model from a previous task') 
            Sherlock_Net=torch.load(previous_task_model_path)
            
        #set model on GPU
    
        if use_gpu:
            Sherlock_Net=Sherlock_Net.cuda()
            
    else:
        checkpoint = torch.load(resume_path)
        Sherlock_Net = checkpoint['model']
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume_path, checkpoint['epoch']))

            

    'SET LEARNING RATE AND WEIGHT DECAY'
    params=train_eval_utils.set_param_learning_rate(Sherlock_Net,lr)

    'TRAIN'
    reg_lambda=float(reg_lambda)
    elastic_caller.finetune_elastic(dset_loaders,Sherlock_Net,params,criterion, epochs,exp_dir,checkpoint,lr,test_iter,reg_lambda,use_multiple_gpu)

def finetune_elastic_inverted(root=None,previous_task_model_path=None,train_data_path=None,test_data_path=None,exp_dir=None,reg_lambda=None,epochs=None,lr=None,test_iter=None,batch=None):
    
    'DEFINE TRAINING PARAMETERS'
    root,previous_task_model_path,train_data_path,test_data_path,exp_dir,reg_lambda,epochs,lr,test_iter,batch=fill_default_vals(root,previous_task_model_path,train_data_path,test_data_path,exp_dir,reg_lambda,epochs,lr,test_iter,batch)
    
    #make your save directory
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
 
    #define LOSS
    criterion = sherlock_model_utils.Fact_Euclidian_Loss()
    print ('Criterion:', type(criterion))

    # in case you have a checkpoint saved in the resume path, the training will start from there
    resume_path = exp_dir+ '/checkpoint.pth.tar'


    'LOAD YOUR DATASET INFORMATION'
    r"""

    df_train and df_test are dataframe holding:
        - image_links: rel_path to each image
        - NLP_links:   rel_path to each NLP representation of each image
        - SPO: fact representation S:subject, P:Predicate, O:Object
        - id : fact label. (each fact has its own Unique label)
        - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

    """
    df_train = pd.read_csv(train_data_path)
    df_test= pd.read_csv(test_data_path)
    
    ##################################################################################################################

    'MAKE YOUR  DATASET'
    # your dataloader will hold
    # images, NLP, ws, wp, wo, labels = data 

    train_dt = dataset_utils.Cst_Dataset(df_train['image_links'], df_train['NLP_links'], df_train['id'], df_train['w_s'] , df_train['w_p'], df_train['w_o'], 
                          root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader,  transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.RandomSizedCrop(224),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std
                                                                        ]))


    test_dt = dataset_utils.Cst_Dataset(df_test['image_links'], df_test['NLP_links'], df_test['id'], df_test['w_s'] , df_test['w_p'], df_test['w_o'] ,
                         root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader, transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                        ]))




    del  df_train, df_test

    # Make your dataset accessible in batches
    dset_loaders = {'train': torch.utils.data.DataLoader(train_dt, batch_size=batch, shuffle=True, num_workers=4),
                    'val':torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=True, num_workers=4)}


    use_gpu = torch.cuda.is_available()  
    print('Training will be done using gpu %s'%use_gpu)
    
    ##################################################################################################################
    
    'LOAD YOUR INITIAL MODEL'
    
    #loading previous model   
    Sherlock_Net=[]
    
    if not os.path.isfile(resume_path):
        checkpoint=[]
        
        #TRAINING ON TASK 1
        if not os.path.isfile(previous_task_model_path):
            #build your Sherlock Net from stratch            
            Sherlock_Net = sherlock_model_utils.build_Sherlock_Net()
            
            #initialize Sherloch Net with VGG16 params
            Sherlock_Net = sherlock_model_utils.initialize_from_VGG(Sherlock_Net, use_gpu)
                 
        
        #FINETUNING
        else:
            #Importing model from previous task.
            print('Loading model from a previous task') 
            Sherlock_Net=torch.load(previous_task_model_path)
        
        #set model on GPU
        if use_gpu:
            Sherlock_Net=Sherlock_Net.cuda()
    else:
        checkpoint = torch.load(resume_path)
        Sherlock_Net = checkpoint['model']
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume_path, checkpoint['epoch']))

    'SET LEARNING RATE AND WEIGHT DECAY'
    params=train_eval_utils.set_param_learning_rate(Sherlock_Net,lr)

    'TRAIN'
    reg_lambda=float(reg_lambda)
    elastic_caller.finetune_elastic_inverted(dset_loaders,Sherlock_Net,params,criterion, epochs,exp_dir,checkpoint,lr,test_iter,reg_lambda)