#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:47:47 2017

@author: raljundi
"""
#importing package
import os
import pdb
import torch
import numpy as np
import scipy.io as sio
from torch.autograd import Variable
from PIL import ImageFile
from torchvision import transforms, models
from utils.sherlock_model_utils import *
ImageFile.LOAD_TRUNCATED_IMAGES = True
######################################################################################################
#TRAIN functions
def exp_lr_scheduler(optimizer, epoch, init_lr=0.000005, lr_decay_epoch=250):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
        print('LR will go down of factor 10 after %s epochs'%lr_decay_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer   

def Cst_exp_lr_scheduler(optimizer, epoch, init_lr=0.000005, lr_decay_epoch=250):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    
    if ((epoch % lr_decay_epoch ==0) & ( epoch !=0)):
        #set your lr multiplier
        lr_mlt = 0.1
        
        #every lr_decay_epochs, each param lr will decrease of a factor 10
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mlt        
     
    return optimizer       

    
def save_checkpoint(state, is_best, save_dir,use_multiple_gpu=0):
    torch.save(state, save_dir + '/checkpoint.pth.tar')
    if is_best:
        model=state['model']
        
        if use_multiple_gpu:
            model=model.module
        torch.save(model,save_dir + '/model_best.pth.tar')

        
#defining layers learning rate gorups
freeze_layers=['net_c.0.weight','net_c.0.bias','net_c.2.weight','net_c.2.bias','net_c.5.weight','net_c.5.bias','net_c.7.weight','net_c.7.bias']

lr1_layers=['net_c.10.weight','net_c.12.weight','net_c.14.weight',                 'net_s_features.0.weight','net_s_features.2.weight','net_s_features.4.weight','net_s_features.7.weight','net_s_features.9.weight','net_s_features.11.weight','net_s_fc.0.weight','net_s_fc.3.weight']

lr2_layers=['net_c.10.bias','net_c.12.bias','net_c.14.bias',                   'net_s_features.0.bias','net_s_features.2.bias','net_s_features.4.bias','net_s_features.7.bias','net_s_features.9.bias','net_s_features.11.bias','net_s_fc.0.bias','net_s_fc.3.bias',                  'net_po_features.0.weight','net_po_features.2.weight','net_po_features.4.weight','net_po_features.7.weight','net_po_features.9.weight','net_po_features.11.weight']

lr4_layers=['net_po_features.0.bias','net_po_features.2.bias','net_po_features.4.bias','net_po_features.7.bias','net_po_features.9.bias','net_po_features.11.bias',]

lr10_layers=['net_s_fc.6.weight','net_po_fc.0.weight','net_po_fc.3.weight','net_po_fc.6.weight']

lr20_layers=['net_s_fc.6.bias','net_po_fc.0.bias','net_po_fc.3.bias','net_po_fc.6.bias']


#defining layers learning rate gorups
model1_freeze_layers=['net_c.0.weight','net_c.0.bias','net_c.2.weight','net_c.2.bias','net_c.5.weight','net_c.5.bias','net_c.7.weight','net_c.7.bias']

model1_lr1_layers=['net_c.10.weight','net_c.12.weight','net_c.14.weight',                 'net_c_features.0.weight','net_c_features.2.weight','net_c_features.4.weight','net_c_features.7.weight','net_c_features.9.weight','net_c_features.11.weight','net_c_fc.0.weight','net_c_fc.3.weight']

#model1_lr2_layers=['net_c.10.bias','net_c.12.bias','net_c.14.bias',                   'net_c_features.0.bias','net_c_features.2.bias','net_c_features.4.bias','net_c_features.7.bias','net_c_features.9.bias','net_c_features.11.bias','net_c_fc.0.bias','net_c_fc.3.bias']
model1_lr4_layers=['net_c.10.bias','net_c.12.bias','net_c.14.bias',                   'net_c_features.0.bias','net_c_features.2.bias','net_c_features.4.bias','net_c_features.7.bias','net_c_features.9.bias','net_c_features.11.bias','net_c_fc.0.bias','net_c_fc.3.bias']
#lr4_layers=['net_po_features.0.bias','net_po_features.2.bias','net_po_features.4.bias','net_po_features.7.bias','net_po_features.9.bias','net_po_features.11.bias',]

model1_lr10_layers=['net_s_fc.0.weight','net_po_fc.0.weight']

model1_lr20_layers=['net_s_fc.0.bias','net_po_fc.0.bias']
def set_param_learning_rate(Sherlock_Net,lr):
  
    
    if isinstance(Sherlock_Net, Model_1):
        return set_model1_param_learning_rate(Sherlock_Net,lr)
    else:
        return set_model2_param_learning_rate(Sherlock_Net,lr)
    
def set_model2_param_learning_rate(Sherlock_Net,lr):
    print('Set per layers learning rate and weights decay')
    #list networks layers in block, according to their needed lr multiplier    
    # assigns weight decay and lr layer by layer
    params = []
    
    for name, param in Sherlock_Net.named_parameters():
     
        name=name.replace('module.','')

        if name in freeze_layers:
            #params += [{'params': [param], 'lr': 0.0*lr, 'weight_decay': 0}]   # the bias will have null weight decay
            print(len(params))
            print(name,'lr=0')
            
        if name in lr1_layers:
            if '.bias' in name:
                params += [{'params': [param], 'lr': 1.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 1.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(1.0*lr))
            
        if name in lr2_layers:
            if '.bias' in name:
                params += [{'params': [param], 'lr': 2.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 2.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(2.0*lr))
            
        if name in lr4_layers:
            if '.bias' in name:
                params += [{'params': [param], 'lr': 4.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 4.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(4.0*lr))
        
        if name in lr10_layers:
            if '.bias' in name:
                params += [{'params': [param], 'lr': 10.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 10.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(10.0*lr))
    
        if name in lr20_layers:
            if '.bias' in name:
                params += [{'params': [param], 'lr': 20.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 20.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(20.0*lr))
    
    return params

########################################################################################

def set_model1_param_learning_rate(Sherlock_Net,lr):
    print('Set per layers learning rate and weights decay')
    #list networks layers in block, according to their needed lr multiplier    
    # assigns weight decay and lr layer by layer
    params = []
    
    for name, param in Sherlock_Net.named_parameters():
     
        name=name.replace('module.','')

        if name in model1_freeze_layers:
            #params += [{'params': [param], 'lr': 0.0*lr, 'weight_decay': 0}]   # the bias will have null weight decay
            print(len(params))
            print(name,'lr=0')
            
        if name in model1_lr1_layers:
            if '.bias' in name:
                params += [{'params': [param], 'lr': 1.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 1.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(1.0*lr))
            

        if name in model1_lr4_layers:
            if '.bias' in name:
                params += [{'params': [param], 'lr': 4.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 4.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(4.0*lr))            

        
        if name in model1_lr10_layers:
            if '.bias' in name:
                params += [{'params': [param], 'lr': 10.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 10.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(10.0*lr))
    
        if name in model1_lr20_layers:
            if '.bias' in name:
                params += [{'params': [param], 'lr': 20.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 20.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(20.0*lr))
    
    return params

########################################################################################

def set_smaller_param_learning_rate(Sherlock_Net,lr):
    print('Set per layers learning rate and weights decay')
    #list networks layers in block, according to their needed lr multiplier    
    # assigns weight decay and lr layer by layer
    params = []
    for name, param in Sherlock_Net.named_parameters():
        
        if name in freeze_layers:
            #params += [{'params': [param], 'lr': 0.0*lr, 'weight_decay': 0}]   # the bias will have null weight decay
            print(len(params))
            print(name,'lr=0')
            
        if name in lr1_layers:
            if '.bias' in name:
                params += [{'params': [param], 'lr': 1.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 1.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(1.0*lr))
            
        if name in lr2_layers:
            if '.bias' in name:
                params += [{'params': [param], 'lr': 2.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 2.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(2.0*lr))
            
        if name in lr4_layers:   #from 4 to 2
            if '.bias' in name:
                params += [{'params': [param], 'lr': 2.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 2.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(2.0*lr))
        
        if name in lr10_layers:   #from 10 to 1
            if '.bias' in name:
                params += [{'params': [param], 'lr': 1.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 1.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(1.0*lr))
    
        if name in lr20_layers:  #from 20 to 2
            if '.bias' in name:
                params += [{'params': [param], 'lr': 2.0*lr, 'weight_decay': 0}]
            else:
                params += [{'params': [param], 'lr': 2.0*lr}]
            print(len(params))
            print(name,'lr=%s'%str(2.0*lr))

    return params



def train_fact_model_2(model, dset_loaders, criterion, optimizer, lr_scheduler, save_dir, num_epochs, test_iter, resume,use_multiple_gpu=None):
    #train your model for "num_epochs" epochs, evaluating the model every "test_iter" epochs
    use_gpu = torch.cuda.is_available()
    
    #initialize loss
    best_val_loss= 100000
    is_best=0
    
    #if your resume path has a checkpoint saved
#    if os.path.isfile(resume):
#        print("=> loading checkpoint '{}'".format(resume))
#        checkpoint = torch.load(resume)
#        
#        #resume trainig: epoch, weights, optimizer
#        start_epoch = checkpoint['epoch']       
#        model.load_state_dict(checkpoint['state_dict'])
#        optimizer.load_state_dict(checkpoint['optimizer'])
#        print("=> loaded checkpoint '{}' (epoch {})"
#              .format(resume, checkpoint['epoch']))
#        del checkpoint
#    
#    #if your resume path has no checkpoint saved    
#    else:
#        #let's start from scratch
#        start_epoch=0
#        print("=> no checkpoint found at '{}'".format(resume))  
    
    if  not resume :
        #let's start from scratch
        start_epoch=0
        print("=> no checkpoint found at ")          
        
    else:

        checkpoint=resume
        start_epoch = checkpoint['epoch'] +1     
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
       
        del checkpoint
    #del fro memory usage
    if use_multiple_gpu:
        #model.features = torch.nn.DataParallel(model.features)
        if hasattr(model,'reg_params'):
            reg_params=model.reg_params
            
            
            model = torch.nn.DataParallel(model).cuda()
            model.reg_params=reg_params
        else:
            
            model = torch.nn.DataParallel(model).cuda()
        #pdb.set_trace()
        
    for epoch in range(start_epoch, num_epochs):
             
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # every "test_iter" epochs, test and save a screenshot of your model.
        if (epoch % test_iter == 0):
            phases=['train', 'val']            
            print('I will evaluate the model and save a snapshot now')
            
        else:
            phases=['train']
            print('Just Train')
            
        for phase in phases:                                  
            running_loss = 0.0  #you will plot the mean loss over all possible batch
                
            #set model in train and eval mode
            if phase =='train':
                optimizer = lr_scheduler(optimizer, epoch) 
                model.train()
            else:
                model.eval()                                   
            
            #process all the data one batch at the time
            for data in dset_loaders[phase]:   
                #print('new batch')#after deadline
                #get your data               
                images, NLP, ws, wp, wo, labels = data    
                
                #get your language target 
                L_S=NLP[:, 0:300]
                L_P=NLP[:, 300:600]
                L_O=NLP[:, 600:900]
          
                if use_gpu:
                    images, L_S, L_P, L_O, ws, wp, wo = Variable(images.cuda()), Variable(L_S.float().cuda(), requires_grad=False), Variable(L_P.float().cuda(), requires_grad=False), Variable(L_O.float().cuda(), requires_grad=False), Variable(ws.cuda(), requires_grad=False), Variable(wp.cuda(), requires_grad=False), Variable(wo.cuda(), requires_grad=False)
                else:
                    images, L_S, L_P, L_O, ws, wp, wo = Variable(images), Variable(L_S.float(), requires_grad=False), Variable(L_P.float(), requires_grad=False), Variable(L_O.float(), requires_grad=False), Variable(ws, requires_grad=False), Variable(wp, requires_grad=False), Variable(wo, requires_grad=False)
                
                #zero_grad
                optimizer.zero_grad() 
                model.zero_grad()
                                 
                #compute output
                V_S, V_PO = model(images) 
                
                
                
                #Now , slice V_PO in V_O and V_P
                
                #1)Probably you could do just a cut of the variable in half
                #V_P = V_PO[:, 0:300]
                #V_O = V_PO[:, 300:600]
                
                #2)To avoid any 'computational graph problem' 
                #Slice the V_PO in half multipling for a 0,1 matrix
                
                #compute 2 diag matrix 300,300 
                one_diag=torch.diag(torch.ones(int(V_PO.size(1)/2))).cuda()
                zero_diag=torch.zeros(300,300).cuda()
                
                #concat them to make 2 slicer 600-600
                mpP=torch.cat((one_diag,zero_diag),0)  # the second 300*300 is an all-zeros matrix , the first 300*300 is a diag matrix (1 over diag, 0 elsewhere)
                mpO=torch.cat((zero_diag,one_diag),0)  # the first 300*300 is an all-zeros matrix , the second 300*300 is a diag matrix (1 over diag, 0 elsewhere)
                
                #multiply the slicer for the V_PO output to get V_P, V_O                
                V_P=torch.mm(V_PO,Variable(mpP,requires_grad=False))
                V_O=torch.mm(V_PO,Variable(mpO,requires_grad=False))
                
                #sanity check: this should be the same as cut the V_PO directly.
                if not torch.equal(V_P.data,V_PO[:,0:300].data):
                    pdb.set_trace()
                    print('****************WARNING NOT CORRECT *********************************')
                if not torch.equal(V_O.data,V_PO[:,300:600].data):
                    pdb.set_trace()
                    print('****************WARNING NOT CORRECT *********************************')    
                
                del images, NLP, labels, data
                
                                   
                #compute Loss
                Loss = criterion(V_S, V_P, V_O, L_S, L_P, L_O, ws, wp, wo)            
                
                if phase == 'train':                
                    # backprog during training 
                    Loss.backward()
                    if hasattr(model,'reg_params'):
                        optimizer.step(model.reg_params)
                    else:
                        optimizer.step()
                                                      
                # update epoch Loss
                running_loss += Loss.data[0]                
                del Loss, V_S, V_P, V_O, V_PO, L_S, L_P, L_O, ws, wp, wo , mpP, mpO, one_diag, zero_diag
                
           
            #print Epoch Loss
            epoch_loss = running_loss / len(dset_loaders[phase])                               
            print('{} Loss:{:.4f} '.format(phase, epoch_loss))  
            
            #DeepCopy and Save the model
            #- if snapshot is the best one saved replace best_model  

            if phase == 'val':  

                #check if your model is the best computed so far
                if epoch_loss < best_val_loss:
                        is_best=1
                        best_val_loss=epoch_loss
                #this line has been commented in order to get consistent results with every thing else        
                #else:
                #    is_best=0
                
                save_checkpoint({
                            'epoch': epoch ,
                            'epoch_loss': epoch_loss,
                            'arch': 'Sherlock_Net',
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'model':model,
                        }, is_best, save_dir,use_multiple_gpu)
                

 #################################################################################################################
#finetuning functions
def finetune_SGD(dset_loaders,model_ft,params,criterion, num_epochs=100,exp_dir='./', resume='',lr=0.0008,test_iter=1000,reg_lambda=10):
   
    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0008, momentum=0.9)


       
    
    optimizer_ft = torch.optim.SGD(params, lr, momentum=0.9, weight_decay=0.0005)
    #optimizer_ft = Elastic_Training.Elastic_SGD(params, lr, momentum=0.9,weight_decay=0.0005)
        
    
    #pdb.set_trace()
   # model_ft = Elastic_Training.train_model(model_ft, criterion, optimizer_ft,exp_lr_scheduler,lr, dset_loaders,dset_sizes,use_gpu,num_epochs,exp_dir,resume)
    model_out = train_fact_model_2(model_ft, dset_loaders, criterion, optimizer_ft, Cst_exp_lr_scheduler, exp_dir, num_epochs, test_iter,resume)

    return model_out   


#Save CV Features function
def save_feat(image_path, V_S, V_P, V_O , root, save_dir, crops=None):
    r"""
    Save in txt file the output of Sherlock_net.
   
    input:
        - Image_path :   name of the sherlock net input (rel_path to image)
        - V_S, V_P, V_O: output of the sherlock net 
        
    output:
        - txt file 
    """
    if crops==1:
        V = torch.cat((V_S.unsqueeze(1), V_P.unsqueeze(1), V_O.unsqueeze(1)),0).cpu().data.numpy()
    
    if crops==10:
        #make feat vector of 900 elements
        V = torch.cat((V_S, V_P, V_O),1).cpu().numpy()
        print ('I am saving:', V.size())
       #V = torch.cat((V_S, V_P, V_O),1).data.cpu().numpy()
    
    
    #get the name for your feat
    path_splits=os.path.split(image_path)
    #name=image_path.replace(root + 'test/6DS_', 'test/6DS_test_CV_')
    #name=image_path.replace(root + 'test/6DS_', 'test/6DS_test_CV_')
    name=path_splits[1]
    name=name.replace('.jpg', '.txt')
    #save it  
    

    np.savetxt(os.path.join(save_dir , name), V) 
    
    #print ('saved:',save_dir + name)
 #use Sherlock_Net model to evaluate your images
def extract_S_features(model, dset_loaders, root, save_dir):
    
        ###################################################
        # 1 CROP CODE
        ###################################################
    
        use_gpu = torch.cuda.is_available()
        print('Evaluate your Sherlock Model')
        print('-' * 10)
        
        #put model in evaluation mode
        model.eval()    
        print ('saving in:',save_dir ) 
        #process all the data one batch at the time
        for data in dset_loaders['val']:             
             
            #get your data
            images, images_paths = data   
            #set model grad to zero 
            model.zero_grad()
            
            #make it a variable
            if use_gpu:
                images =  Variable(images.cuda(), requires_grad=False)
            else:
                images = Variable(images, requires_grad=False)
            
            #compute output
            layer_name='4'
            Sfeat= extract_features_Sbranch(model,images,layer_name)
            
            for im in range(len(Sfeat)):                
               
                           
                
                #get its name
                image_path = images_paths[im]
                            
                #get its name
                
              
                #save it
            
                save_feat_before_fc(image_path,Sfeat[im] , root, save_dir,crops=1 )
            
            del Sfeat
               
            del images_paths
        return   
    
def extract_PO_features(model, dset_loaders, root, save_dir):
    
        ###################################################
        # 1 CROP CODE
        ###################################################
    
        use_gpu = torch.cuda.is_available()
        print('Evaluate your Sherlock Model')
        print('-' * 10)
        
        #put model in evaluation mode
        model.eval()    
        print ('saving in:',save_dir ) 
        #process all the data one batch at the time
        for data in dset_loaders['val']:             
             
            #get your data
            images, images_paths = data   
            #set model grad to zero 
            model.zero_grad()
            
            #make it a variable
            if use_gpu:
                images =  Variable(images.cuda(), requires_grad=False)
            else:
                images = Variable(images, requires_grad=False)
            
            #compute output
            layer_name='4'
            POfeat= extract_features_PObranch(model,images,layer_name)
            
            for im in range(len(POfeat)):                
               
                           
                
                #get its name
                image_path = images_paths[im]
                            
                #get its name
                
              
                #save it
            
                save_feat_before_fc(image_path,POfeat[im] , root, save_dir,crops=1 )
            
            del POfeat
               
            del images_paths
        return   
#
#save features before the fully connected layer
def save_feat_before_fc(image_path, feat, root, save_dir, crops=None):
    r"""
    Save in txt file the output of Sherlock_net.
   
    input:
        - Image_path :   name of the sherlock net input (rel_path to image)
        - V_S, V_P, V_O: output of the sherlock net 
        
    output:
        - txt file 
    """
    if crops==1:
        V = (feat).cpu().data.numpy()
    
    
    
    #get the name for your feat
    path_splits=os.path.split(image_path)
    #name=image_path.replace(root + 'test/6DS_', 'test/6DS_test_CV_')
    #name=image_path.replace(root + 'test/6DS_', 'test/6DS_test_CV_')
    name=path_splits[1]
    name=name.replace('.jpg', '.txt')
    #save it  
    

    np.savetxt(os.path.join(save_dir , name), V) 
    

    
#use Sherlock_Net model to evaluate your images
def eval_fact_model_mat(model, dset_loaders, root, save_dir):
    
        ###################################################
        # 1 CROP CODE
        ###################################################
    
        use_gpu = torch.cuda.is_available()
        print('Evaluate your Sherlock Model')
        print('-' * 10)
        
        #put model in evaluation mode
        model.eval()    
        print ('saving in:',save_dir ) 
        #process all the data one batch at the time
        c=0
        for data in dset_loaders['val']:
            #get your data
            images, images_paths = data   
            #set model grad to zero 
            model.zero_grad()
            
            #make it a variable
            if use_gpu:
                images =  Variable(images.cuda(), requires_grad=False)
            else:
                images = Variable(images, requires_grad=False)
            
            #compute output
            try:
                V_S, V_PO = model(images)
            except: 
                V_S, V_PO,x,y= model(images)
                del x,y
            
            #merge V_S, V_PO in a feat
           
            batch_feats=torch.cat([V_S.data, V_PO.data],1)
            
            #merge batch_feat in the output
            if c==0:
                feats = batch_feats
                c +=1
            else:
                feats = torch.cat([feats,batch_feats],0)  #append the new batch to the output
                c +=1
                print('batch %s'%str(c))

            del V_S, V_PO, batch_feats    
            del images_paths
       
        #np.save(save_dir, feats.cpu().numpy())
        feats=feats.cpu().numpy()
        print(save_dir)
        np.save(save_dir, feats)
        sio.savemat(save_dir + '.mat', {'XE': feats})

        return
    

    


#use Sherlock_Net model to evaluate your images
def eval_fact_model(model, dset_loaders, root, save_dir):
    
        ###################################################
        # 1 CROP CODE
        ###################################################
    
        use_gpu = torch.cuda.is_available()
        print('Evaluate your Sherlock Model')
        print('-' * 10)
        
        #put model in evaluation mode
        model.eval()    
        print ('saving in:',save_dir ) 
        #process all the data one batch at the time
        for data in dset_loaders['val']:             
             
            #get your data
            images, images_paths = data   
            #set model grad to zero 
            model.zero_grad()
            
            #make it a variable
            if use_gpu:
                images =  Variable(images.cuda(), requires_grad=False)
            else:
                images = Variable(images, requires_grad=False)
            
            #compute output
            try:
                V_S, V_PO = model(images)
            except: 
                V_S, V_PO,x,y= model(images)
                del x,y
                
            for im in range(len(V_S)):                
                #get image CV_feat
                V_S_im = V_S[im, :]
                V_P_im = V_PO[im, 0:300]
                V_O_im = V_PO[im, 300:600]
                           
                
                #get its name
                image_path = images_paths[im]
                
                #save it
                
                save_feat(image_path, V_S_im, V_P_im, V_O_im, root, save_dir, crops=1)
                del V_S_im, V_P_im, V_O_im
               
            del images_paths
        return

    
    
def eval_fact_model_10_crop(model, dset_loaders, root, save_dir):
    
        ###################################################
        # 10 CROP CODE
        ###################################################
    
        use_gpu = torch.cuda.is_available()
        print('Evaluate your Sherlock Model')
        print('-' * 10)
        
        #put model in evaluation mode
        model.eval()    
        pdb.set_trace()
        #process all the data one batch at the time
        for data in dset_loaders['val']:             
             
            #get your data
            images, images_paths = data   
            #set model grad to zero 
            model.zero_grad()
            
            #make it a variable
            if use_gpu:
                images =  Variable(images.cuda(), requires_grad=False)
            else:
                images = Variable(images, requires_grad=False)
            
            #compute output
            V_S, V_PO = model(images)
            
            for im in range(len(V_S)):                
                #get image CV_feat
                V_S_im = V_S[im, :]
                V_P_im = V_PO[im, 0:300]
                V_O_im = V_PO[im, 300:600]
                           
                
                #get its name
                image_path = images_paths[im]
                
                #save it
                
                save_feat(image_path, V_S_im, V_P_im, V_O_im, root, save_dir, crops=1)
                del V_S_im, V_P_im, V_O_im
                
            del images_paths
        return
    
def sanitycheck(model):
    name1='net_po_features.11.weight'
    name2='net_s_features.11.weight'
    for name, param in model.named_parameters():
            #w=torch.FloatTensor(param.size()).zero_()
            
           
            if param in model.reg_params:
                if name1==name:
                    p1=param
                if name2==name:
                    p2=param   
                
            
           
    return   p1,p2   
################################################################################
#ECCV
def load_model(resume_path,previous_task_model_path, use_gpu,use_multiple_gpu):    
    #loading previous model   
    Sherlock_Net=[]

    if not os.path.isfile(resume_path):
        checkpoint=[]
       #TRAINING ON TASK 1     
        if not os.path.isfile(previous_task_model_path):
            #build your Sherlock Net from stratch            
            #Sherlock_Net = sherlock_model_utils.build_Sherlock_Net_with_before_fc_output()
            Sherlock_Net = sherlock_model_utils_cvae.build_Sherlock_Net_with_before_fc_output_v2()
            
            #initialize Sherloch Net with VGG16 params
            #Sherlock_Net = sherlock_model_utils.initialize_from_VGG(Sherlock_Net, use_gpu)
            Sherlock_Net = sherlock_model_utils_cvae.initialize_from_VGG_v2(Sherlock_Net, use_gpu)       
        #FINETUNING
        else:
            #Importing model from previous task.
            print('Loading model from a previous task')
            checkpoint=torch.load(previous_task_model_path)
            try:
                Sherlock_Net = checkpoint['model']
            except:
                Sherlock_Net=checkpoint
            del checkpoint
            checkpoint=[]
            
        #set model on GPU
        if use_gpu:
            Sherlock_Net=Sherlock_Net.cuda()
    else:
        checkpoint = torch.load(resume_path)
        Sherlock_Net = checkpoint['model']
        if use_multiple_gpu:
            reg_params=Sherlock_Net.reg_params
            Sherlock_Net=Sherlock_Net.module
            Sherlock_Net = torch.nn.DataParallel(Sherlock_Net)
            Sherlock_Net.reg_params=reg_params

        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume_path, checkpoint['epoch']))
    return Sherlock_Net, checkpoint


def eval_fact_model_mat(model, dset_loaders, root, save_dir):
        feats=eval_fact_model_npy(model, dset_loaders, root, save_dir)
        print(save_dir)
        np.save(save_dir, feats)
        sio.savemat(save_dir + '.mat', {'XE': feats})

        return

def eval_fact_model_npy(model, dset_loaders, root, save_dir):
    
        ###################################################
        # 1 CROP CODE
        ###################################################
    
        use_gpu = torch.cuda.is_available()
        print('Evaluate your Sherlock Model')
        print('-' * 10)
        
        #put model in evaluation mode
        model.eval()    
        print ('saving in:',save_dir ) 
        #process all the data one batch at the time
        c=0
        for data in dset_loaders['val']:             
             
            #get your data
            images, images_paths = data   
            #set model grad to zero 
            model.zero_grad()
            
            #make it a variable
            if use_gpu:
                images =  Variable(images.cuda(), requires_grad=False)
            else:
                images = Variable(images, requires_grad=False)
            
            #compute output
            #try model1
            try:
                V_S, V_PO = model(images)
            except: 
                #try model2
                try:
                    V_S, V_PO,x,y= model(images)
                    del x,y
                #try model3
                except:
                    V_S, V_P, V_O, x,y,z = model(images)
                    V_PO = torch.cat([V_P, V_O],1)
                    del x,y,z
            
            #merge V_S, V_PO in a feat
            batch_feats=torch.cat([V_S.data, V_PO.data],1)
            
            #merge batch_feat in the output
            if c==0:
                feats = batch_feats
                c +=1
            else:
                feats = torch.cat([feats,batch_feats],0)  #append the new batch to the output
                c +=1
                #print('batch %s'%str(c))

            del V_S, V_PO, batch_feats    
            del images_paths
       
        #np.save(save_dir, feats.cpu().numpy())
        feats=feats.cpu().numpy()
        
        return feats
