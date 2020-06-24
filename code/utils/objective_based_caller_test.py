#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:11:26 2017

@author: raljundi
"""
import os
import sys
from Fact_learning_utils.Objective_gradient_utils import Objective_based_Training
from Fact_learning_utils import train_eval_utils
from Fact_learning_utils import dataset_utils
import torch
from torch.autograd import Variable
import pdb
import pandas as pd
from torchvision import transforms, models



def finetune_objective(dset_loaders,model_ft,params,criterion, num_epochs=100,exp_dir='./', resume='',lr=0.0008,test_iter=1000,reg_lambda=1,use_multiple_gpu=None):
    
    #defining the optimizer
    optimizer_ft = Objective_based_Training.Weight_Regularized_SGD(params, lr, momentum=0.9,weight_decay=0.0005)
        
    
    #pdb.set_trace()
       # model_ft = Elastic_Training.train_model(model_ft, criterion, optimizer_ft,exp_lr_scheduler,lr, dset_loaders,dset_sizes,use_gpu,num_epochs,exp_dir,resume)
    model_out = train_eval_utils.train_fact_model_2(model_ft, dset_loaders, criterion, optimizer_ft, train_eval_utils.Cst_exp_lr_scheduler, exp_dir, num_epochs, test_iter,resume,use_multiple_gpu=use_multiple_gpu)
    
    return model_out   
########################################################################################################

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
########################

def update_objective_based_weights(root,model_ft,batch,use_gpu,norm='L2',  reg_sets=['data_info/B1_train.cvs']):
    
    #reg_params=Objective_based_Training.initialize_reg_params(model_ft)
    #model_ft.reg_params=reg_params
    #====================get data loaders============
    dset_loaders=[]
    for data_path in reg_sets:
    
        df_data = pd.read_csv(root+data_path)

        ##################################################################################################################

        'MAKE YOUR  DATASET'
        # your dataloader will hold
        # images, NLP, ws, wp, wo, labels = data 




        data_dt = dataset_utils.Cst_Dataset(df_data['image_links'], df_data['NLP_links'], df_data['id'], df_data['w_s'] , df_data['w_p'], df_data['w_o'] ,
                         root, image_loader=dataset_utils.pil_loader, NLP_feat_loader = dataset_utils.NLP_loader, transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                        ]))




        # Make your dataset accessible in batches
        dset_loader=  torch.utils.data.DataLoader(data_dt, batch_size=batch, shuffle=False, num_workers=4)
        
        dset_loaders.append(dset_loader)
        
    #==============================================
    
    optimizer_ft = Objective_based_Training.Objective_After_SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
    #exp_dir='/esat/monkey/raljundi/pytorch/CUB11f_hebbian_finetuned'
    
    if norm=='L2':
        print('********************objective with L2 norm***************')
        model_ft = compute_importance_l2(model_ft, optimizer_ft,train_eval_utils.Cst_exp_lr_scheduler, dset_loaders,use_gpu)
    else:
        model_ft = compute_importance(model_ft, optimizer_ft,train_eval_utils.Cst_exp_lr_scheduler, dset_loaders,use_gpu)

    return model_ft


def compute_importance_l2(model, optimizer, lr_scheduler,dset_loaders,use_gpu):
    
    print('dictionary length'+str(len(dset_loaders)))
          
    epoch=1
    optimizer = lr_scheduler(optimizer, epoch)
    model.eval()  # Set model to training mode so we get the gradient
    
    
    index=0
    for dset_loader in dset_loaders:
        for data in dset_loader:

            # zero the parameter gradients
            # forward
            #code special for  sherlock model
            #get your data          

            images, NLP, ws, wp, wo, labels = data 
            
            del  NLP, ws, wp, wo


            if use_gpu:
                images = Variable(images.cuda())

            else:
                images = Variable(images)

            #zero_grad
            optimizer.zero_grad() 
            model.zero_grad()

            #compute output
            V_S, V_PO = model(images) 
            V_S_zeros=torch.zeros(V_S.size())
            V_PO_zeros=torch.zeros(V_PO.size())
            
            if use_gpu:
                V_S_zeros=V_S_zeros.cuda()
                V_PO_zeros=V_PO_zeros.cuda()
                
            #compute the L2 norm of output    
            V_S_zeros=Variable(V_S_zeros,requires_grad=False)
            V_PO_zeros=Variable(V_PO_zeros,requires_grad=False)
            loss = torch.nn.MSELoss(size_average=False)

            output = loss(V_S,V_S_zeros) +loss(V_PO,V_PO_zeros) 

            #end of special sherlock model code
            output.backward()
            del V_S_zeros,V_PO_zeros
            #print('step')
            optimizer.step(model.reg_params,index,labels.size(0))
            del images,loss,output,labels
            index+=1
            print('batch #',index)
    return model
###################################################

def compute_importance( model,optimizer, lr_scheduler,dset_loaders,use_gpu):
    print('dictoinary length'+str(len(dset_loaders)))
    
   
    epoch=1
    optimizer = lr_scheduler(optimizer, epoch)
    model.eval()  # Set model to training mode so we get the gradient


    # Iterate over data.
    index=0
    for dset_loader in dset_loaders:
        for data in dset_loader:

            # zero the parameter gradients


            # forward
            #code special for  sherlock model
            #get your data          

            images, NLP, ws, wp, wo, labels = data    
            del  NLP, ws, wp, wo


            if use_gpu:
                images = Variable(images.cuda())

            else:
                images = Variable(images)

            #zero_grad
            optimizer.zero_grad() 
            model.zero_grad()

            #compute output
            V_S, V_PO = model(images) 
            V_S_zeros=torch.zeros(V_S.size())
            V_PO_zeros=torch.zeros(V_PO.size())
            if use_gpu:
                V_S_zeros=V_S_zeros.cuda()
                V_PO_zeros=V_PO_zeros.cuda()
            V_S_zeros=Variable(V_S_zeros,requires_grad=False)
            V_PO_zeros=Variable(V_PO_zeros,requires_grad=False)
            #loss = torch.nn.MSELoss()

            #output = loss(V_S,V_S_zeros) +loss(V_PO,V_PO_zeros) 

            #end of special sherlock model code
            # ones=torch.ones(outputs.size()).cuda()
            #V_S.backward(torch.ones(V_S.size()).cuda(),retain_graph=True)
            V_S.backward(torch.ones(V_S.size()).cuda(),retain_variables=True)
            V_PO.backward(torch.ones(V_PO.size()).cuda())
            del V_S_zeros,V_PO_zeros
            #print('step')
            optimizer.step(model.reg_params,index,labels.size(0))
            del images,labels
            index+=1
            print('batch #',index)
        return model
    
    