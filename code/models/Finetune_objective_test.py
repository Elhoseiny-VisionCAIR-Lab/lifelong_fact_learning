
#!/usr/bin/env python3

"""
Created on Wed Sep 20 19:07:53 2017

@author: raljundi& fbabilon
"""


#adding optimizer packages
import sys

# Implement Sherlock Paper in Pytorch-
##################################################################################################################
import pandas as pd
import pdb
import os
import collections
import torch
import numpy as np
from torchvision import transforms, models
from utils import train_eval_utils
from utils import sherlock_model_utils
from utils import dataset_utils

##WARNING YOU ARE IMPORTING THE TEST VERSION
from utils import objective_based_caller_test

from utils.Objective_gradient_utils import Objective_based_Training

def fill_default_vals(root, previous_task_model_path, train_data_path, test_data_path, exp_dir, reg_lambda, norm,reg_sets, epochs, lr, test_iter, batch):
    func_params=[]
    print('-' * 40)
    print('YOUR PARAMETERS')
    print('-' * 40)
    
    if root is None:
        root= '/users/visics/fbabilon/Sherlock/export_Sherlock/'
    print('the root directory is ',root) 
        
    if reg_lambda is None:
        reg_lambda=1
    print('the regularizer multipler value is ',reg_lambda)
        
    if norm is None:
        norm=''
        print('the objective is the output of the Sherlock model')       
    else:
        print('the objective is the L2  of output of the Sherlock model')
     
    if reg_sets is None:
        reg_sets=['data_info/B1_train.csv']
    print('the objective regulizer is using data from ',' '.join(reg_sets))
   
    
    if exp_dir is None:    
        exp_dir=root+'pytorch_models/B2_objective_based_all'+'/reg_'+str(reg_lambda)+'_'+norm+'/'
        
    print('the exporting directory is ',exp_dir)
        
    if previous_task_model_path is None:      
        previous_task_model_path= root+'pytorch_models/B1_elastic/'    
        previous_task_model_path=previous_task_model_path+'model_best.pth.tar'
    print('the previous task model is ',previous_task_model_path)
        
    if train_data_path is None:
        train_data_path=root + 'data_info/B2_train.csv'
    print('the training data path is ',train_data_path)
        
    if test_data_path is None:
        test_data_path=root + 'data_info/B2_test.csv'
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
    
    print('-' * 40)
    print('')    
    return (root, previous_task_model_path, train_data_path, test_data_path, exp_dir, reg_lambda, norm, reg_sets, epochs, lr, test_iter, batch)
def finetune_objective(root=None,previous_task_model_path=None,train_data_path=None,test_data_path=None,exp_dir=None,reg_lambda=None, norm=None, reg_sets=None, epochs=None,lr=None,test_iter=None,batch=None,use_multiple_gpu=None,b1=True):
    
    root, previous_task_model_path, train_data_path, test_data_path, exp_dir, reg_lambda, norm, reg_sets, epochs, lr, test_iter, batch=fill_default_vals(root,previous_task_model_path, train_data_path, test_data_path, exp_dir, reg_lambda, norm, reg_sets, epochs,lr,test_iter,batch)
    #define root , save_dir and log_file



    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)


    'DEFINE TRAINING'

    #define LOSS
    criterion = sherlock_model_utils.Fact_Euclidian_Loss()


    print ('Using epochs:', epochs, 'Criterion:', type(criterion))

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
                                                                        ]))#changed to the new pytroch version


    test_dt = dataset_utils.Cst_Dataset(df_test['image_links'], df_test['NLP_links'], df_test['id'], df_test['w_s'] , df_test['w_p'], df_test['w_o'] ,
                         root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader, transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                        ]))




    

    # Make your dataset accessible in batches
    dset_loaders = {'train': torch.utils.data.DataLoader(train_dt, batch_size=batch, shuffle=True, num_workers=4),
                    'val':torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=True, num_workers=4)}


    use_gpu = torch.cuda.is_available()
  
    print('Training will be done using gpu %s'%use_gpu)

    #loading previous model
    objective_init_file=exp_dir+'/init_model.pth.tar'
    
    
  
    if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path)
            Sherlock_Net = checkpoint['model']

            print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume_path, checkpoint['epoch']))
    else:
        checkpoint=[]
    
        print('Loading previous task model') 
        Sherlock_Net=torch.load(previous_task_model_path)
        print('loaded')
            
    #Sherlock_Net = previous_task_model_path 
    #initialize reg_params to zero
    #it doesn't create parameters for freezed layers
       
        reg_params=Objective_based_Training.initialize_reg_params(Sherlock_Net,train_eval_utils.freeze_layers)
        Sherlock_Net.reg_params=reg_params

        #use both val and train
        print('Entering test functions') 
        if use_multiple_gpu:
            reg_batch=35
        else:
            reg_batch=batch
        if b1:
            reg_batch=1
        Sherlock_Net=objective_based_caller_test.update_objective_based_weights(root,Sherlock_Net,reg_batch,use_gpu,norm, reg_sets)
        
        
        sanitycheck(Sherlock_Net)
        Sherlock_Net.reg_params['lambda']=float(reg_lambda)

    if use_gpu:
        Sherlock_Net=Sherlock_Net.cuda()
    del  df_train, df_test
    'SET LEARNING RATE AND WEIGHT DECAY'
    params=train_eval_utils.set_param_learning_rate(Sherlock_Net,lr)
    print('Start finetuning objective' )
    #TRAIN 
  
    objective_based_caller_test.finetune_objective(dset_loaders,Sherlock_Net,params,criterion, epochs,exp_dir,checkpoint,lr,test_iter,use_multiple_gpu=use_multiple_gpu)

def finetune_objective_cumulative(root=None,previous_task_model_path=None,train_data_path=None,test_data_path=None,exp_dir=None,reg_lambda=None, norm=None, reg_sets=None, epochs=None,lr=None,test_iter=None,batch=None,use_multiple_gpu=None,b1=True):
    
    root, previous_task_model_path, train_data_path, test_data_path, exp_dir, reg_lambda, norm, reg_sets, epochs, lr, test_iter, batch=fill_default_vals(root,previous_task_model_path, train_data_path, test_data_path, exp_dir, reg_lambda, norm, reg_sets, epochs,lr,test_iter,batch)
    #define root , save_dir and log_file



    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)


    'DEFINE TRAINING'

    #define LOSS
    criterion = sherlock_model_utils.Fact_Euclidian_Loss()


    print ('Using epochs:', epochs, 'Criterion:', type(criterion))

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
                                                                        ]))#changed to the new pytroch version


    test_dt = dataset_utils.Cst_Dataset(df_test['image_links'], df_test['NLP_links'], df_test['id'], df_test['w_s'] , df_test['w_p'], df_test['w_o'] ,
                         root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader, transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                        ]))




    

    # Make your dataset accessible in batches
    dset_loaders = {'train': torch.utils.data.DataLoader(train_dt, batch_size=batch, shuffle=True, num_workers=4),
                    'val':torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=True, num_workers=4)}


    use_gpu = torch.cuda.is_available()
  
    print('Training will be done using gpu %s'%use_gpu)

    #loading previous model
    objective_init_file=exp_dir+'/init_model.pth.tar'
    
    
  
    if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path)
            Sherlock_Net = checkpoint['model']
            if use_multiple_gpu:
                reg_params=Sherlock_Net.reg_params
                Sherlock_Net=Sherlock_Net.module
                Sherlock_Net = torch.nn.DataParallel(Sherlock_Net)
                Sherlock_Net.reg_params=reg_params
            print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume_path, checkpoint['epoch']))
    else:
        checkpoint=[]
    
        print('Loading previous task model') 
        Sherlock_Net=torch.load(previous_task_model_path)
        print('loaded')
            
    #Sherlock_Net = previous_task_model_path 
    #initialize reg_params to zero
    #it doesn't create parameters for freezed layers
       
        reg_params=Objective_based_Training.initialize_store_reg_params(Sherlock_Net,train_eval_utils.freeze_layers)
        Sherlock_Net.reg_params=reg_params

        #use both val and train
        print('Entering test functions') 
        if use_multiple_gpu:
            reg_batch=35
        else:
            reg_batch=batch
        if b1:
            reg_batch=1
        Sherlock_Net=objective_based_caller_test.update_objective_based_weights(root,Sherlock_Net,reg_batch,use_gpu,norm, reg_sets)
        #pdb.set_trace()
        reg_params=Objective_based_Training.accumelate_reg_params(Sherlock_Net,train_eval_utils.freeze_layers)
        sanitycheck(Sherlock_Net)
        Sherlock_Net.reg_params=reg_params   
        #pdb.set_trace()
        Sherlock_Net.reg_params['lambda']=float(reg_lambda)

    if use_gpu:
        Sherlock_Net=Sherlock_Net.cuda()
    del  df_train, df_test
    'SET LEARNING RATE AND WEIGHT DECAY'
    params=train_eval_utils.set_param_learning_rate(Sherlock_Net,lr)
    print('Start finetuning objective' )
    #TRAIN 
   
    objective_based_caller_test.finetune_objective(dset_loaders,Sherlock_Net,params,criterion, epochs,exp_dir,checkpoint,lr,test_iter,use_multiple_gpu=use_multiple_gpu)

#accumelate omega over tasks instead of computing it each time on the new test sets

def finetune_objective_subtractoin(root=None,previous_task_model_path=None,train_data_path=None,test_data_path=None,exp_dir=None,reg_lambda=None, norm=None, reg_sets=None, epochs=None,lr=None,test_iter=None,batch=None,use_multiple_gpu=None):
    
    root, previous_task_model_path, train_data_path, test_data_path, exp_dir, reg_lambda, norm, reg_sets, epochs, lr, test_iter, batch=fill_default_vals(root,previous_task_model_path, train_data_path, test_data_path, exp_dir, reg_lambda, norm, reg_sets, epochs,lr,test_iter,batch)
    #define root , save_dir and log_file



    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)


    'DEFINE TRAINING'

    #define LOSS
    criterion = sherlock_model_utils.Fact_Euclidian_Loss()


    print ('Using epochs:', epochs, 'Criterion:', type(criterion))

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
                                                                        ]))#changed to the new pytroch version


    test_dt = dataset_utils.Cst_Dataset(df_test['image_links'], df_test['NLP_links'], df_test['id'], df_test['w_s'] , df_test['w_p'], df_test['w_o'] ,
                         root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader, transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                        ]))




    

    # Make your dataset accessible in batches
    dset_loaders = {'train': torch.utils.data.DataLoader(train_dt, batch_size=batch, shuffle=True, num_workers=4),
                    'val':torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=True, num_workers=4)}


    use_gpu = torch.cuda.is_available()
  
    print('Training will be done using gpu %s'%use_gpu)

    #loading previous model
    objective_init_file=exp_dir+'/init_model.pth.tar'
    
    
  
    if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path)
            Sherlock_Net = checkpoint['model']
            if use_multiple_gpu:
                reg_params=Sherlock_Net.reg_params
                Sherlock_Net=Sherlock_Net.module
                Sherlock_Net = torch.nn.DataParallel(Sherlock_Net)
                Sherlock_Net.reg_params=reg_params
            print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume_path, checkpoint['epoch']))
    else:
        checkpoint=[]
    
        print('Loading previous task model') 
        Sherlock_Net=torch.load(previous_task_model_path)
        print('loaded')
    #Sherlock_Net = previous_task_model_path 
    #initialize reg_params to zero
    #it doesn't create parameters for freezed layers
        sub_reg_sets=[reg_sets[len(reg_sets)-1]]
        acc_reg_sets=reg_sets[0:len(reg_sets)-1]

        reg_params=Objective_based_Training.initialize_store_reg_params(Sherlock_Net,train_eval_utils.freeze_layers)
        Sherlock_Net.reg_params=reg_params

        #use both val and train
        print('Entering test functions') 
        if use_multiple_gpu:
            reg_batch=35
        else:
            reg_batch=batch
        Sherlock_Net=objective_based_caller_test.update_objective_based_weights(root,Sherlock_Net,reg_batch,use_gpu,norm, acc_reg_sets)
        #pdb.set_trace()
        reg_params=Objective_based_Training.accumelate_reg_params(Sherlock_Net,train_eval_utils.freeze_layers)
        sanitycheck(Sherlock_Net)
        Sherlock_Net.reg_params=reg_params  
        reg_params=Objective_based_Training.initialize_store_reg_params(Sherlock_Net,train_eval_utils.freeze_layers)
        Sherlock_Net.reg_params=reg_params
        sanitycheck(Sherlock_Net)
        #use both val and train
        print('Entering test functions') 
        if use_multiple_gpu:
            reg_batch=35
        else:
            reg_batch=batch
   
        Sherlock_Net=objective_based_caller_test.update_objective_based_weights(root,Sherlock_Net,reg_batch,use_gpu,norm, sub_reg_sets)
        #pdb.set_trace()
        reg_params=Objective_based_Training.subtract_reg_params(Sherlock_Net,train_eval_utils.freeze_layers)
        sanitycheck(Sherlock_Net)
        Sherlock_Net.reg_params=reg_params   
        Sherlock_Net.reg_params['lambda']=float(reg_lambda)

    if use_gpu:
        Sherlock_Net=Sherlock_Net.cuda()
    del  df_train, df_test
    'SET LEARNING RATE AND WEIGHT DECAY'
    params=train_eval_utils.set_param_learning_rate(Sherlock_Net,lr)
    print('Start finetuning objective' )
    #TRAIN 

    objective_based_caller_test.finetune_objective(dset_loaders,Sherlock_Net,params,criterion, epochs,exp_dir,checkpoint,lr,test_iter,use_multiple_gpu=use_multiple_gpu)

#compute mean of  omegas over tasks instead of computing it each time on the new test sets

def finetune_objective_mean_cumulative(root=None,previous_task_model_path=None,train_data_path=None,test_data_path=None,exp_dir=None,reg_lambda=None, norm=None, reg_sets=None, epochs=None,lr=None,test_iter=None,batch=None,num_of_tasks=2):
    
    root, previous_task_model_path, train_data_path, test_data_path, exp_dir, reg_lambda, norm, reg_sets, epochs, lr, test_iter, batch=fill_default_vals(root,previous_task_model_path, train_data_path, test_data_path, exp_dir, reg_lambda, norm, reg_sets, epochs,lr,test_iter,batch)
    #define root , save_dir and log_file



    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)


    'DEFINE TRAINING'

    #define LOSS
    criterion = sherlock_model_utils.Fact_Euclidian_Loss()


    print ('Using epochs:', epochs, 'Criterion:', type(criterion))

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
                                                                        ]))#changed to the new pytroch version


    test_dt = dataset_utils.Cst_Dataset(df_test['image_links'], df_test['NLP_links'], df_test['id'], df_test['w_s'] , df_test['w_p'], df_test['w_o'] ,
                         root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader, transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                        ]))




    

    # Make your dataset accessible in batches
    dset_loaders = {'train': torch.utils.data.DataLoader(train_dt, batch_size=batch, shuffle=True, num_workers=4),
                    'val':torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=True, num_workers=4)}


    use_gpu = torch.cuda.is_available()
  
    print('Training will be done using gpu %s'%use_gpu)

    #loading previous model
    objective_init_file=exp_dir+'/init_model.pth.tar'
    
    
  
    if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path)
            Sherlock_Net = checkpoint['model']

            print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume_path, checkpoint['epoch']))
    else:
        checkpoint=[]
    
        print('Loading previous task model') 
        Sherlock_Net=torch.load(previous_task_model_path)
        print('loaded')
            
    #Sherlock_Net = previous_task_model_path 
    #initialize reg_params to zero
    #it doesn't create parameters for freezed layers
        
        reg_params=Objective_based_Training.initialize_store_reg_params(Sherlock_Net,train_eval_utils.freeze_layers)
        Sherlock_Net.reg_params=reg_params

        #use both val and train
        print('Entering test functions') 

        Sherlock_Net=objective_based_caller_test.update_objective_based_weights(root,Sherlock_Net,batch,use_gpu,norm, reg_sets)
      
        reg_params=Objective_based_Training.accumelate_avg_reg_params(Sherlock_Net,train_eval_utils.freeze_layers,num_of_tasks)
        #pdb.set_trace()
        sanitycheck(Sherlock_Net)
        Sherlock_Net.reg_params=reg_params   
        Sherlock_Net.reg_params['lambda']=float(reg_lambda)

    if use_gpu:
        Sherlock_Net=Sherlock_Net.cuda()
    del  df_train, df_test
    'SET LEARNING RATE AND WEIGHT DECAY'
    params=train_eval_utils.set_param_learning_rate(Sherlock_Net,lr)
    print('Start finetuning objective' )
    #TRAIN 
    objective_based_caller_test.finetune_objective(dset_loaders,Sherlock_Net,params,criterion, epochs,exp_dir,checkpoint,lr,test_iter)

def sanitycheck(model):
    for name, param in model.named_parameters():
            #w=torch.FloatTensor(param.size()).zero_()
            print (name)
            if param in model.reg_params:
            
                reg_param=model.reg_params.get(param)
                omega=reg_param.get('omega')
                
                print('omega max is',omega.max())
                print('omega min is',omega.min())
                print('omega mean is',omega.mean())