#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:40:38 2017

@author: raljundi
"""
import sys
import os
import pdb
from Fact_learning_utils.Elastic_utils import Elastic_Training
from Fact_learning_utils import train_eval_utils
import torch
def finetune_elastic(dset_loaders,model_ft,params,criterion, num_epochs=100,exp_dir='./', resume='',lr=0.0008,test_iter=1000,reg_lambda=1,use_multiple_gpu=None):
    
    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0008, momentum=0.9)
    if not resume:
        
        if not hasattr(model_ft, 'reg_params'):
            reg_params=Elastic_Training.initialize_reg_params(model_ft)
            print('initializing the regulization parameters')
        
        else:          
            reg_params=Elastic_Training.update_reg_params(model_ft)
            #add invert omega flag
            print('update the regulization parameters')
            
        reg_params['lambda']=reg_lambda        
        model_ft.reg_params=reg_params
        
    #if not resume    
    optimizer_ft = Elastic_Training.Elastic_SGD(params, lr, momentum=0.9,weight_decay=0.0005)
    model_out = train_eval_utils.train_fact_model_2(model_ft, dset_loaders, criterion, optimizer_ft, train_eval_utils.Cst_exp_lr_scheduler, exp_dir, num_epochs, test_iter,resume,use_multiple_gpu)

    return model_out   

def finetune_elastic_inverted(dset_loaders,model_ft,params,criterion, num_epochs=100,exp_dir='./', resume='',lr=0.0008,test_iter=1000,reg_lambda=1):   
    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0008, momentum=0.9)
    if not resume:
        
        if not hasattr(model_ft, 'reg_params'):
            
            reg_params=Elastic_Training.initialize_reg_params(model_ft)
        
            print('initializing the regulization parameters')
        else:
            
            reg_params=Elastic_Training.update_reg_params(model_ft)
            model_ft.reg_params=reg_params
            
            model_ft=normalize(model_ft)
            model_ft=invert_omega(model_ft)
            model_ft=normalize(model_ft)
            reg_params=model_ft.reg_params
            #torch.save(reg_params,'reg_params')
            #add invert omega flag
            print('update the regulization parameters')
            
        reg_params['lambda']=reg_lambda        
        model_ft.reg_params=reg_params
        
    #if not resume
    
    optimizer_ft = Elastic_Training.Elastic_SGD(params, lr, momentum=0.9,weight_decay=0.0005)
        
  
    model_out = train_eval_utils.train_fact_model_2(model_ft, dset_loaders, criterion, optimizer_ft, train_eval_utils.Cst_exp_lr_scheduler, exp_dir, num_epochs, test_iter,resume)

    return model_out  
def invert_omega(model):
    for name, param in model.named_parameters():
            #w=torch.FloatTensor(param.size()).zero_()
            print (name)
            if param in model.reg_params:
            
                reg_param=model.reg_params.get(param)
                omega=reg_param.get('omega')
                omega=omega+1e-4
                new_omega=1/omega
                reg_param['omega']=new_omega
                model.reg_params[param]=reg_param
                print('omega max is',new_omega.max())
                print('omega min is',new_omega.min())
                print('omega mean is',new_omega.mean())
    return model

def normalize(model):
    omega_max=0
    omega_min=1e3
    #getting max and min
    for name, param in model.named_parameters():
        #w=torch.FloatTensor(param.size()).zero_()
        print (name)
        if param in model.reg_params:
            zero=torch.FloatTensor(param.size()).zero_()

            reg_param=model.reg_params.get(param)
            omega=reg_param.get('omega')

            if not omega.equal(zero.cuda()):
                if omega.max()>omega_max:
                    omega_max=omega.max()
                if omega.min()<omega_min:
                    omega_min=omega.min()
        #normalizing
    for name, param in model.named_parameters():
        if param in model.reg_params:
            #w=torch.FloatTensor(param.size()).zero_()
            print (name)
            zero=torch.FloatTensor(param.size()).zero_()

            reg_param=model.reg_params.get(param)
            omega=reg_param.get('omega')

            if not omega.equal(zero.cuda()):
                omega=(omega-omega_min).div(omega_max-omega_min)
            reg_param['omega']=omega    
    return model