#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:49:55 2017

@author: raljundi
"""

"""
Created on Tue Sep 19 11:55:57 2017

@author: fbabilon
"""
import sys
#sys.path.append("/users/visics/raljundi/Code/MyOwnCode/Pytorch") 
import os

import numpy as np
import pdb
from PIL import Image
import matplotlib.pyplot as plt
import scipy.spatial.distance as spd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional  as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


import numbers

from PIL import ImageFile
from Fact_learning_utils.Elastic_utils import Elastic_Training
from Fact_learning_utils.Objective_gradient_utils import Objective_based_Training

ImageFile.LOAD_TRUNCATED_IMAGES = True

#################################################################################################################


# Loss functions 
def Fact_Euclidian_distance(V_S, V_P, V_O, L_S, L_P, L_O, ws, wp, wo, size_average=True): 
    r"""
    Computes the euclidian distance between the Computer Vision feat.(V) and the NLP feat.(L).
    Computing the distance for each part of the fact S,P,O separately and then join them togethr using the boolean flag(w).
    
    In case a fact has no Predicate or Object, the relative flag will be zero (w=0) and the related pairwise distance will give no contribution.
    
    output:
        - size_average=1: retuns the mean fact euclidian distance over the given batch
        - size_average=0: returns the sum fact euclidian distance over the given batch
   
    """
    use_gpu = torch.cuda.is_available()	
	 
    if use_gpu:
        Loss =  ( ws.float().unsqueeze(1).cuda() * F.pairwise_distance(V_S, L_S) + 
                  wp.float().unsqueeze(1).cuda() * F.pairwise_distance(V_P, L_P) + 
                  wo.float().unsqueeze(1).cuda() * F.pairwise_distance(V_O, L_O) )

    else:
        Loss =  ( ws.float().unsqueeze(1).cuda() * F.pairwise_distance(V_S, L_S) + 
                  wp.float().unsqueeze(1).cuda() * F.pairwise_distance(V_P, L_P) + 
                  wo.float().unsqueeze(1).cuda() * F.pairwise_distance(V_O, L_O) )
    
    if size_average:
        return Loss.mean()
        
    else:
        return Loss.sum()
    

class Fact_Euclidian_Loss(nn.MSELoss):
    r"""Creates a criterion that measures the Euclidian Distance between a fact representation
    in Computer Vision and in Natural Language Processing. It will process a batch at the time. 
    
    inputs:
        - V_S, V_P, V_O = Computer Vision features
        - L_S, L_P, L_O = Natural Language Processing features
        - ws, wp, wo = boolean flag
    
    outputs:
        - batch Loss value. 

    """
    def __init__(self):
        super(Fact_Euclidian_Loss, self).__init__()
       
    def forward(self, V_S, V_P, V_O, L_S, L_P, L_O, ws, wp, wo):        
        return Fact_Euclidian_distance(V_S, V_P, V_O, L_S, L_P, L_O, ws, wp, wo, size_average=self.size_average)   

#############################################################################################################
################ GEN SAMPLES LOSS
# Loss functions 
def GEN_Euclidian_distance(V_S,L_S, ws, size_average=True): 
    r"""
    Computes the euclidian distance between the Computer Vision feat.(V) and the NLP feat.(L).
    Computing the distance for each part of the fact S,P,O separately and then join them togethr using the boolean flag(w).
    
    In case a fact has no Predicate or Object, the relative flag will be zero (w=0) and the related pairwise distance will give no contribution.
    
    output:
        - size_average=1: retuns the mean fact euclidian distance over the given batch
        - size_average=0: returns the sum fact euclidian distance over the given batch
   
    """
    use_gpu = torch.cuda.is_available()	
	 
    if use_gpu:
        Loss =  ( ws.float().unsqueeze(1).cuda() * F.pairwise_distance(V_S, L_S) )
                

    else:
        Loss =  ( ws.float().unsqueeze(1).cuda() * F.pairwise_distance(V_S, L_S) )
    
    if size_average:
        return Loss.mean()
        
    else:
        return Loss.sum()
    

class GEN_Euclidian_Loss(nn.MSELoss):
    r"""Creates a criterion that measures the Euclidian Distance between a fact representation
    in Computer Vision and in Natural Language Processing. It will process a batch at the time. 
    
    inputs:
        - V_S = Computer Vision features
        - L_S = Natural Language Processing features
        - ws= boolean flag
    
    outputs:
        - batch Loss value. 

    """
    def __init__(self):
        super(GEN_Euclidian_Loss, self).__init__()
       
    def forward(self, V_S, L_S, ws):        
        return GEN_Euclidian_distance(V_S, L_S, ws, size_average=self.size_average)   
################ END OF GEN SAMPLES LOSS
# Build Sherlock_Net functions
class Model_2(nn.Module):
    r"""
    Sherlock_Net:
        input: batch of images (x) 
        output: s_fc and po_fc 
    """
    def __init__(self, net_c, net_s_feat, net_s_fc, net_po_feat, net_po_fc):
        super(Model_2, self).__init__()
        
        # define each piece of network 
        self.net_c = net_c # segment 1 from VGG until features[17]
        
        self.net_s_features = net_s_feat  # segment 2 from VGG 
        self.net_po_features = net_po_feat # segment 2 from VGG 
        
        self.net_s_fc = net_s_fc
        self.net_po_fc = net_po_fc
        
       #define forward.  
    def forward(self, x):
        #get your batch of images
        x = self.net_c(x)
        
        #make yout input pass through subject branch 
        s_feat = self.net_s_features(x)
        s_feat = s_feat.view(s_feat.size(0), -1)
        s_fc   = self.net_s_fc(s_feat) 
        
        #make yout input pass through object-predicate branch 
        po_feat = self.net_po_features(x) 
        po_feat = po_feat.view(po_feat.size(0), -1)
        po_fc = self.net_po_fc(po_feat)
                
        return s_fc, po_fc
class Model_1(nn.Module):
    r"""
    Sherlock_Net:
        input: batch of images (x) 
        output: s_fc and po_fc 
    """
    #TO BE UPDATED
    def __init__(self,net_c, net_c_feat, net_c_fc, net_s_fc, net_po_fc):
        super(Model_1, self).__init__()
        
        # define each piece of network 
        self.net_c = net_c # segment 1 from VGG until features[17]
        
        self.net_c_features = net_c_feat  # segment 2 from VGG 

        
        self.net_c_fc = net_c_fc
        self.net_s_fc = net_s_fc
        self.net_po_fc = net_po_fc
        
       #define forward.  
    def forward(self, x):
        #get your batch of images
        x = self.net_c(x)
        
        #make yout input pass through subject branch 
        #c for common
        c_feat = self.net_c_features(x)
        c_feat = c_feat.view(c_feat.size(0), -1)
        c_fc   = self.net_c_fc(c_feat) 
        s_fc   = self.net_s_fc(c_fc) 
        po_fc   = self.net_po_fc(c_fc) 
        
                
        return s_fc, po_fc    
#it also returns the features before the last fully connected layer
class Model_2_beforefc_feat(nn.Module):
    r"""
    Sherlock_Net:
        input: batch of images (x) 
        output: s_fc and po_fc 
    """
    def __init__(self, net_c, net_s_feat, net_s_fc, net_po_feat, net_po_fc):
        super(Model_2_beforefc_feat, self).__init__()
        
        # define each piece of network 
        self.net_c = net_c # segment 1 from VGG until features[17]
        
        self.net_s_features = net_s_feat  # segment 2 from VGG 
        self.net_po_features = net_po_feat # segment 2 from VGG 
        
        self.net_s_fc = net_s_fc
        self.net_po_fc = net_po_fc
        
       #define forward.  
    def forward(self, x):
        #get your batch of images
        shared_input = self.net_c(x)
        
        #make yout input pass through subject branch 
        s_feat = self.net_s_features(shared_input)
        s_feat = s_feat.view(s_feat.size(0), -1)
        #s_fc   = self.net_s_fc(s_feat) 
        
        x = s_feat  
        #layer before dropout and the last fc
        layer_name='4'
        for name, module in self.net_s_fc._modules.items():
            
            x = module(x)
            
            if name ==layer_name:
               
                s_before_nlp_pro=x
                
        s_fc=x #should matcj 207
        
           
        #make yout input pass through object-predicate branch 
        po_feat = self.net_po_features(shared_input) 
        po_feat = po_feat.view(po_feat.size(0), -1)
        #po_fc = self.net_po_fc(po_feat)
        x=po_feat
        for name, module in self.net_po_fc._modules.items():



            x = module(x)
            
            if name ==layer_name:
               
                po_before_nlp_pro=x
                
        po_fc=x
                        
        return s_fc, po_fc,po_before_nlp_pro,s_before_nlp_pro
    
def extract_features_Sbranch(sherlock_model, x, layer_name):
        #get your batch of images
        x = sherlock_model.net_c(x)
       
        #make yout input pass through subject branch 
        s_feat = sherlock_model.net_s_features(x)
        s_feat = s_feat.view(s_feat.size(0), -1)
        if 0:     
            s_fc   = sherlock_model.net_s_fc(s_feat) 
        
            #make yout input pass through object-predicate branch 
            po_feat = self.net_po_features(x) 
            po_feat = po_feat.view(po_feat.size(0), -1)
            po_fc = self.net_po_fc(po_feat)
        
        
        output=None
        x=s_feat
       
        
        for name, module in sherlock_model.net_s_fc._modules.items():



            x = module(x)
            
            if name ==layer_name:
               
                output=x
                break
                
        return output           

def extract_features_PObranch(sherlock_model, x, layer_name):
        #get your batch of images
        x = sherlock_model.net_c(x)
       
        #make yout input pass through subject branch 
        po_feat = sherlock_model.net_po_features(x) 
        po_feat = po_feat.view(po_feat.size(0), -1)
        
        
       
      
        
        
        output=None
        x=po_feat
       
        
        for name, module in sherlock_model.net_po_fc._modules.items():



            x = module(x)
            
            if name ==layer_name:
               
                output=x
                break
                
        return output         

def make_layers(params, ch): 
    layers = []
    channels = ch
    k=0
    for p in params:
            conv2d = nn.Conv2d(channels, p, kernel_size=3, stride=1, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            channels = p
            k= k+1           
            if ((( k % 2 ==0) and k !=6) or k ==7):
                MaxPool = nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
                layers += [MaxPool]
    return nn.Sequential(*layers) 

def make_layers_2_feat(params, ch, d): 
    layers = []
    channels = ch
    k=0
    for p in params:
            conv2d = nn.Conv2d(channels, p, kernel_size=3, stride=1, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            channels = p
            k= k+1           
            if (k % d ==0) :
                MaxPool = nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
                layers += [MaxPool]
                                         
    return nn.Sequential(*layers) 

def make_layers_2_fc(net):   
    
    if net == 's':
        return nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 300),
        )
        
    if net == 'po':
        return  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 600),
        )
#shared fc layers for model1   
def make_layers_1_fc(net):   
    

    return nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
    )
#the last fc layers for model1  where the s and po branches split up   
def make_disjoint_layers_1_fc(net):   
    

    if net == 's':
        return nn.Sequential(
           
            nn.Linear(4096, 300),
        )
        
    if net == 'po':
        return  nn.Sequential(

            nn.Linear(4096, 600),
        )
    
# BUILD SHERLOCK NET MULTIPLE OUTPUT V2
def make_layers_2_fc_a(net):   
    
    if net == 's':
        return nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )
        
    if net == 'po':
        return  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )
    
def make_layers_2_fc_b(net):   
    
    if net == 's':
        return nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 300),
        )
        
    if net == 'po':
        return  nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 600),
        )

    
    
    
####################################################################################################
# BUILD SHERLOCK NET MULTIPLE OUTPUT V2
#it also returns the features before the last fully connected layer
class Model_2_beforefc_feat_v2(nn.Module):
    r"""
    Sherlock_Net:
        input: batch of images (x) 
        output: s_fc and po_fc 

        
       #define forward.  
    def forward(self, x):
        #get your batch of images
        x = self.net_c(x)
        
        #make yout input pass through subject branch 
        s_feat = self.net_s_features(x)
        s_feat = s_feat.view(s_feat.size(0), -1)
        s_fc   = self.net_s_fc(s_feat) 
        
        #make yout input pass through object-predicate branch 
        po_feat = self.net_po_features(x) 
        po_feat = po_feat.view(po_feat.size(0), -1)
        po_fc = self.net_po_fc(po_feat)
                
    """
    def __init__(self, net_c, net_s_feat, net_s_fc_a, net_s_fc_b, net_po_feat, net_po_fc_a, net_po_fc_b):
        super(Model_2_beforefc_feat_v2, self).__init__()
        
        # define each piece of network 
        self.net_c = net_c # segment 1 from VGG until features[17]
        
        self.net_s_features = net_s_feat  # segment 2 from VGG 
        self.net_po_features = net_po_feat # segment 2 from VGG 
        
        self.net_s_fc_a = net_s_fc_a
        self.net_po_fc_a = net_po_fc_a
        
        self.net_s_fc_b = net_s_fc_b
        self.net_po_fc_b = net_po_fc_b
        
       #define forward.  
    def forward(self, x):
        #get your batch of images
        shared_input = self.net_c(x)
        
        #make yout input pass through subject branch 
        s_feat = self.net_s_features(shared_input)
        s_feat = s_feat.view(s_feat.size(0), -1)
        s_fc_a   = self.net_s_fc_a(s_feat) #s_before_nlp_pro
        #s_fc_a = s_feat.view(s_fc_a.size(0), -1) #dunno if needed
        s_fc_b   = self.net_s_fc_b(s_fc_a)
        
        #make yout input pass through object-predicate branch 
        po_feat = self.net_po_features(shared_input) 
        po_feat = po_feat.view(po_feat.size(0), -1)
        po_fc_a = self.net_po_fc_a(po_feat)  #po_before_nlp_pro
        po_fc_b = self.net_po_fc_b(po_fc_a)
        
        return s_fc_b, po_fc_b, po_fc_a, po_fc_a
        
        r"""
        x = s_feat  
        #layer before dropout and the last fc
        layer_name='4'
        for name, module in self.net_s_fc._modules.items():
            
            x = module(x)
            
            if name ==layer_name:
               
                s_before_nlp_pro=x
        pdb.set_trace()        
        s_fc=x #should matcj 207
        
           
        #make yout input pass through object-predicate branch 
        po_feat = self.net_po_features(shared_input) 
        po_feat = po_feat.view(po_feat.size(0), -1)
        #po_fc = self.net_po_fc(po_feat)
        x=po_feat
        for name, module in self.net_po_fc._modules.items():



            x = module(x)
            
            if name ==layer_name:
               
                po_before_nlp_pro=x
                
        po_fc=x
                        
        return s_fc, po_fc,po_before_nlp_pro,s_before_nlp_pro
        """
    
def extract_features_Sbranch(sherlock_model, x, layer_name):
        #get your batch of images
        x = sherlock_model.net_c(x)
       
        #make yout input pass through subject branch 
        s_feat = sherlock_model.net_s_features(x)
        s_feat = s_feat.view(s_feat.size(0), -1)
        if 0:     
            s_fc   = sherlock_model.net_s_fc(s_feat) 
        
            #make yout input pass through object-predicate branch 
            po_feat = self.net_po_features(x) 
            po_feat = po_feat.view(po_feat.size(0), -1)
            po_fc = self.net_po_fc(po_feat)
        
        
        output=None
        x=s_feat
       
        
        for name, module in sherlock_model.net_s_fc._modules.items():



            x = module(x)
            
            if name ==layer_name:
               
                output=x
                break
                
        return output           

def extract_features_PObranch(sherlock_model, x, layer_name):
        #get your batch of images
        x = sherlock_model.net_c(x)
       
        #make yout input pass through subject branch 
        po_feat = sherlock_model.net_po_features(x) 
        po_feat = po_feat.view(po_feat.size(0), -1)
        
        
       
      
        
        
        output=None
        x=po_feat
       
        
        for name, module in sherlock_model.net_po_fc._modules.items():



            x = module(x)
            
            if name ==layer_name:
               
                output=x
                break
                
        return output         



#####################################################################################################
def build_Sherlock_Net():
	'BUILD YOUR SHERLOCK NETWORK'
	r"""
	Sherlock_Net has:
	    
	    - Initial Common Branch:
		- net_c - taking from conv_1 to Maxpool_3 of VGG_16
	    
	    - Subject Branch:
		- net_s_feat - taking from conv_8 to Maxpool_5 of VGG_16 
		- net_s_fc   - taking from fc_1 to fc3 of VGG_16 
		
	    -Predicate and Object Branch:
		- net_po_feat - taking from conv_8 to Maxpool_5 of VGG_16 
		- net_po_fc   - taking from fc_1 to fc3 of VGG_16 
	"""
	# Build you Network
	#first build your slots separately. 
	#convert it into a function
	net_c  = make_layers([64, 64, 128, 128, 256, 256, 256], 3)

	net_po_feat = make_layers_2_feat([512, 512, 512, 512, 512, 512], 256, 3)
	net_s_feat  = make_layers_2_feat([512, 512, 512, 512, 512, 512], 256, 3)

	net_po_fc = make_layers_2_fc('po')
	net_s_fc  = make_layers_2_fc('s')
	print('I am here') 
	#net build your network. As default pytorch measure, everything is initialized randomly
	Sherlock_Net = Model_2(net_c, net_s_feat, net_s_fc, net_po_feat, net_po_fc)
	print('Building Sherlock Net')    
	return Sherlock_Net
#build the model1 of Sherlock net
def build_model1_Sherlock_Net():
	'BUILD YOUR SHERLOCK NETWORK'
	r"""
	Sherlock_Net has:
	    
	    - Initial Common Branch:
		- net_c - taking from conv_1 to Maxpool_3 of VGG_16
	    
	    - Subject Branch:
		- net_s_feat - taking from conv_8 to Maxpool_5 of VGG_16 
		- net_s_fc   - taking from fc_1 to fc3 of VGG_16 
		
	    -Predicate and Object Branch:
		- net_po_feat - taking from conv_8 to Maxpool_5 of VGG_16 
		- net_po_fc   - taking from fc_1 to fc3 of VGG_16 
	"""
	# Build you Network
	#first build your slots separately. 
	#convert it into a function
	net_c  = make_layers([64, 64, 128, 128, 256, 256, 256], 3)
	net_c_feat  = make_layers_2_feat([512, 512, 512, 512, 512, 512], 256, 3)
	#pdb.set_trace()
	net_c_fc  = make_layers_1_fc('c')
	net_po_fc = make_disjoint_layers_1_fc('po')
	net_s_fc  = make_disjoint_layers_1_fc('s')
	
	
	print('I am here') 
	#net build your network. As default pytorch measure, everything is initialized randomly
	Sherlock_Net = Model_1(net_c, net_c_feat, net_c_fc, net_s_fc, net_po_fc)
	print('Building Sherlock Net')    
	return Sherlock_Net
#it returns additional output before the projection
def build_Sherlock_Net_with_before_fc_output():
	'BUILD YOUR SHERLOCK NETWORK'
	r"""
	Sherlock_Net has:
	    
	    - Initial Common Branch:
		- net_c - taking from conv_1 to Maxpool_3 of VGG_16
	    
	    - Subject Branch:
		- net_s_feat - taking from conv_8 to Maxpool_5 of VGG_16 
		- net_s_fc   - taking from fc_1 to fc3 of VGG_16 
		
	    -Predicate and Object Branch:
		- net_po_feat - taking from conv_8 to Maxpool_5 of VGG_16 
		- net_po_fc   - taking from fc_1 to fc3 of VGG_16 
	"""
	# Build you Network
	#first build your slots separately. 
	#convert it into a function
	net_c  = make_layers([64, 64, 128, 128, 256, 256, 256], 3)

	net_po_feat = make_layers_2_feat([512, 512, 512, 512, 512, 512], 256, 3)
	net_s_feat  = make_layers_2_feat([512, 512, 512, 512, 512, 512], 256, 3)

	net_po_fc = make_layers_2_fc('po')
	net_s_fc  = make_layers_2_fc('s')
	print('I am here') 
	#net build your network. As default pytorch measure, everything is initialized randomly
	Sherlock_Net = Model_2_beforefc_feat(net_c, net_s_feat, net_s_fc, net_po_feat, net_po_fc)
	print('Building Sherlock Net with additional before projection output')    
	return Sherlock_Net


def build_Sherlock_Net_with_before_fc_output_v2():
	'BUILD YOUR SHERLOCK NETWORK'
	r"""
	Sherlock_Net has:
	    
	    - Initial Common Branch:
		- net_c - taking from conv_1 to Maxpool_3 of VGG_16
	    
	    - Subject Branch:
		- net_s_feat - taking from conv_8 to Maxpool_5 of VGG_16 
		- net_s_fc   - taking from fc_1 to fc3 of VGG_16 
		
	    -Predicate and Object Branch:
		- net_po_feat - taking from conv_8 to Maxpool_5 of VGG_16 
		- net_po_fc   - taking from fc_1 to fc3 of VGG_16 
	"""
	# Build you Network
	#first build your slots separately. 
	#convert it into a function
	net_c  = make_layers([64, 64, 128, 128, 256, 256, 256], 3)

	net_po_feat = make_layers_2_feat([512, 512, 512, 512, 512, 512], 256, 3)
	net_s_feat  = make_layers_2_feat([512, 512, 512, 512, 512, 512], 256, 3)

	net_po_fc_a = make_layers_2_fc_a('po')
	net_s_fc_a  = make_layers_2_fc_a('s')
    
	net_po_fc_b = make_layers_2_fc_b('po')
	net_s_fc_b  = make_layers_2_fc_b('s')
	print('I am here') 
	#net build your network. As default pytorch measure, everything is initialized randomly
	Sherlock_Net = Model_2_beforefc_feat_v2(net_c, net_s_feat, net_s_fc_a, net_s_fc_b, net_po_feat, net_po_fc_a, net_po_fc_b)
	print('Building Sherlock Net with additional before projection output')    
	return Sherlock_Net

######################################################################################################
# Import VGG architecture for initializzation
def initialize_from_VGG(Sherlock_Net, use_gpu):

    'INITIALIZE SHERLOCK MODEL WITH A PRE-TRAINED CHECKPOINT'
    '''
    # load the pre-trained weights
    Sherlock_Net_pretrained= Model_2(net_c, net_s_feat, net_s_fc, net_po_feat, net_po_fc)
    path = save_dir+ 'model_best.pth.tar'
    checkpoint = torch.load(path)
    #import weights from pretrained model
    Sherlock_Net_pretrained.load_state_dict(checkpoint['state_dict'])
    #set_model_on_GPU
    if use_gpu:             
       Sherlock_Net_pretrained=Sherlock_Net_pretrained.cuda(0) 

    print 'recover training from Sherlock pretrained model '
    '''

    'INITIALIZE SHERLOCK MODEL WITH VGG16 PARAMS'

    # Import VGG architecture for initializzation
    model= models.vgg16(pretrained=True)
    if use_gpu:            #set_model_on_GPU
       model=model.cuda(0) 


    # initialize net_c            
    for index in range(16):
          Sherlock_Net.net_c._modules[str(index)] = model.features._modules[str(index)]

    print('WARNING: .clone() removed')
    # initialize net_po_feat, net_s_feat
    for index in range(17,30):
          if hasattr( model.features._modules[str(index)], 'weight'):
             Sherlock_Net.net_po_features._modules[str(index-17)].weight.data = model.features._modules[str(index)].weight.data#.clone()
             Sherlock_Net.net_po_features._modules[str(index-17)].bias.data = model.features._modules[str(index)].bias.data#.clone()
             Sherlock_Net.net_s_features._modules[str(index-17)].weight.data  = model.features._modules[str(index)].weight.data#.clone()
             Sherlock_Net.net_s_features._modules[str(index-17)].bias.data  = model.features._modules[str(index)].bias.data#.clone()
          else:
              Sherlock_Net.net_po_features._modules[str(index-17)] = model.features._modules[str(index)]
              Sherlock_Net.net_s_features._modules[str(index-17)]  = model.features._modules[str(index)]


    # initialize net_po_fc, net_s_fc. 
    for index in range(0,5):  #You are leaving initialized randomly the last fully connected layer (6)
          if hasattr( model.classifier._modules[str(index)], 'weight'):
              Sherlock_Net.net_po_fc._modules[str(index)].weight.data = model.classifier._modules[str(index)].weight.data#.clone()
              Sherlock_Net.net_po_fc._modules[str(index)].bias.data = model.classifier._modules[str(index)].bias.data#.clone()
              Sherlock_Net.net_s_fc._modules[str(index)].weight.data  = model.classifier._modules[str(index)].weight.data#.clone()
              Sherlock_Net.net_s_fc._modules[str(index)].bias.data  = model.classifier._modules[str(index)].bias.data#.clone()
          else:
              Sherlock_Net.net_po_fc._modules[str(index)] = model.classifier._modules[str(index)]
              Sherlock_Net.net_s_fc._modules[str(index)]  = model.classifier._modules[str(index)]


    #initializing the last layers
    Sherlock_Net.net_s_fc._modules['6'].weight.data.normal_(0, 0.005)
    Sherlock_Net.net_s_fc._modules['6'].bias.data.fill_(1)
    #m.bias.data.fill_(0)
    Sherlock_Net.net_po_fc._modules['6'].weight.data.normal_(0, 0.005)
    Sherlock_Net.net_po_fc._modules['6'].bias.data.fill_(1)

    print('Sherlock_Net Initialized with VGG parameters') 
    
    del model
    return Sherlock_Net
#====================================model2=======================================================
# Import VGG architecture for initializzation
def initialize_model2_from_VGG(Sherlock_Net, use_gpu):

    'INITIALIZE SHERLOCK MODEL WITH A PRE-TRAINED CHECKPOINT'
    '''
    # load the pre-trained weights
    Sherlock_Net_pretrained= Model_2(net_c, net_s_feat, net_s_fc, net_po_feat, net_po_fc)
    path = save_dir+ 'model_best.pth.tar'
    checkpoint = torch.load(path)
    #import weights from pretrained model
    Sherlock_Net_pretrained.load_state_dict(checkpoint['state_dict'])
    #set_model_on_GPU
    if use_gpu:             
       Sherlock_Net_pretrained=Sherlock_Net_pretrained.cuda(0) 

    print 'recover training from Sherlock pretrained model '
    '''

    'INITIALIZE SHERLOCK MODEL WITH VGG16 PARAMS'

    # Import VGG architecture for initializzation
    model= models.vgg16(pretrained=True)
    if use_gpu:            #set_model_on_GPU
       model=model.cuda(0) 


    # initialize net_c            
    for index in range(16):
          Sherlock_Net.net_c._modules[str(index)] = model.features._modules[str(index)]

    print('.clone()')
    # initialize net_po_feat, net_s_feat
    for index in range(17,30):
          if hasattr( model.features._modules[str(index)], 'weight'):
             Sherlock_Net.net_po_features._modules[str(index-17)].weight.data = model.features._modules[str(index)].weight.data.clone()
             Sherlock_Net.net_po_features._modules[str(index-17)].bias.data = model.features._modules[str(index)].bias.data.clone()
             Sherlock_Net.net_s_features._modules[str(index-17)].weight.data  = model.features._modules[str(index)].weight.data.clone()
             Sherlock_Net.net_s_features._modules[str(index-17)].bias.data  = model.features._modules[str(index)].bias.data.clone()
          else:
              Sherlock_Net.net_po_features._modules[str(index-17)] = model.features._modules[str(index)]
              Sherlock_Net.net_s_features._modules[str(index-17)]  = model.features._modules[str(index)]


    # initialize net_po_fc, net_s_fc. 
    for index in range(0,5):  #You are leaving initialized randomly the last fully connected layer (6)
          if hasattr( model.classifier._modules[str(index)], 'weight'):
              Sherlock_Net.net_po_fc._modules[str(index)].weight.data = model.classifier._modules[str(index)].weight.data.clone()
              Sherlock_Net.net_po_fc._modules[str(index)].bias.data = model.classifier._modules[str(index)].bias.data.clone()
              Sherlock_Net.net_s_fc._modules[str(index)].weight.data  = model.classifier._modules[str(index)].weight.data.clone()
              Sherlock_Net.net_s_fc._modules[str(index)].bias.data  = model.classifier._modules[str(index)].bias.data.clone()
          else:
              Sherlock_Net.net_po_fc._modules[str(index)] = model.classifier._modules[str(index)]
              Sherlock_Net.net_s_fc._modules[str(index)]  = model.classifier._modules[str(index)]


    #initializing the last layers
    Sherlock_Net.net_s_fc._modules['6'].weight.data.normal_(0, 0.005)
    Sherlock_Net.net_s_fc._modules['6'].bias.data.fill_(1)
    #m.bias.data.fill_(0)
    Sherlock_Net.net_po_fc._modules['6'].weight.data.normal_(0, 0.005)
    Sherlock_Net.net_po_fc._modules['6'].bias.data.fill_(1)

    print('Sherlock_Net model2 Initialized with VGG parameters') 
    
    del model
    return Sherlock_Net


#=====================================model1=======================================================
# Import VGG architecture for initialization
def initialize_model1_from_VGG(Sherlock_Net, use_gpu):

    'INITIALIZE SHERLOCK MODEL WITH A PRE-TRAINED CHECKPOINT'
    '''
    # load the pre-trained weights
    Sherlock_Net_pretrained= Model_2(net_c, net_s_feat, net_s_fc, net_po_feat, net_po_fc)
    path = save_dir+ 'model_best.pth.tar'
    checkpoint = torch.load(path)
    #import weights from pretrained model
    Sherlock_Net_pretrained.load_state_dict(checkpoint['state_dict'])
    #set_model_on_GPU
    if use_gpu:             
       Sherlock_Net_pretrained=Sherlock_Net_pretrained.cuda(0) 

    print 'recover training from Sherlock pretrained model '
    '''

    'INITIALIZE SHERLOCK MODEL WITH VGG16 PARAMS'

    # Import VGG architecture for initializzation
    model= models.vgg16(pretrained=True)
    if use_gpu:            #set_model_on_GPU
       model=model.cuda(0) 


    # initialize net_c            
    for index in range(16):
          Sherlock_Net.net_c._modules[str(index)] = model.features._modules[str(index)]


    # initialize net_c_feat
    for index in range(17,30):
          if hasattr( model.features._modules[str(index)], 'weight'):
             Sherlock_Net.net_c_features._modules[str(index-17)].weight.data = model.features._modules[str(index)].weight.data.clone()
             Sherlock_Net.net_c_features._modules[str(index-17)].bias.data = model.features._modules[str(index)].bias.data.clone()
            
          else:
              Sherlock_Net.net_c_features._modules[str(index-17)] = model.features._modules[str(index)]
              


    # initialize net_c_fc
    for index in range(0,5):  #You are leaving initialized randomly the last fully connected layer (6)
          if hasattr( model.classifier._modules[str(index)], 'weight'):
              Sherlock_Net.net_c_fc._modules[str(index)].weight.data = model.classifier._modules[str(index)].weight.data.clone()
              Sherlock_Net.net_c_fc._modules[str(index)].bias.data = model.classifier._modules[str(index)].bias.data.clone()

          else:
              Sherlock_Net.net_c_fc._modules[str(index)] = model.classifier._modules[str(index)]
              Sherlock_Net.net_c_fc._modules[str(index)]  = model.classifier._modules[str(index)]

    #initializing the last layers
    Sherlock_Net.net_s_fc._modules['0'].weight.data.normal_(0, 0.005)
    Sherlock_Net.net_s_fc._modules['0'].bias.data.fill_(1)
    #m.bias.data.fill_(0)
    Sherlock_Net.net_po_fc._modules['0'].weight.data.normal_(0, 0.005)
    Sherlock_Net.net_po_fc._modules['0'].bias.data.fill_(1)

    print('Sherlock_Net Initialized with VGG parameters') 
    
    del model
    return Sherlock_Net
###################################################################################################
# Import VGG architecture for initializzation BEFORE NLP OUTPUT
def initialize_from_VGG_v2(Sherlock_Net, use_gpu):

    'INITIALIZE SHERLOCK MODEL WITH A PRE-TRAINED CHECKPOINT'
    '''
    # load the pre-trained weights
    Sherlock_Net_pretrained= Model_2(net_c, net_s_feat, net_s_fc, net_po_feat, net_po_fc)
    path = save_dir+ 'model_best.pth.tar'
    checkpoint = torch.load(path)
    #import weights from pretrained model
    Sherlock_Net_pretrained.load_state_dict(checkpoint['state_dict'])
    #set_model_on_GPU
    if use_gpu:             
       Sherlock_Net_pretrained=Sherlock_Net_pretrained.cuda(0) 

    print 'recover training from Sherlock pretrained model '
    '''

    'INITIALIZE SHERLOCK MODEL WITH VGG16 PARAMS'

    # Import VGG architecture for initializzation
    model= models.vgg16(pretrained=True)
    if use_gpu:            #set_model_on_GPU
       model=model.cuda(0) 


    # initialize net_c            
    for index in range(16):
          Sherlock_Net.net_c._modules[str(index)] = model.features._modules[str(index)]


    # initialize net_po_feat, net_s_feat
    for index in range(17,30):
          if hasattr( model.features._modules[str(index)], 'weight'):
             Sherlock_Net.net_po_features._modules[str(index-17)].weight.data = model.features._modules[str(index)].weight.data.clone()
             Sherlock_Net.net_po_features._modules[str(index-17)].bias.data = model.features._modules[str(index)].bias.data.clone()
             Sherlock_Net.net_s_features._modules[str(index-17)].weight.data  = model.features._modules[str(index)].weight.data.clone()
             Sherlock_Net.net_s_features._modules[str(index-17)].bias.data  = model.features._modules[str(index)].bias.data.clone()
          else:
              Sherlock_Net.net_po_features._modules[str(index-17)] = model.features._modules[str(index)]
              Sherlock_Net.net_s_features._modules[str(index-17)]  = model.features._modules[str(index)]


    # initialize net_po_fc_a, net_s_fc_a.
    
    #You are leaving initialized randomly the last fully connected layer (6) --> net_po_fc_b, net_s_fc_b
    
    for index in range(0,4):  
          if hasattr( model.classifier._modules[str(index)], 'weight'):
              Sherlock_Net.net_po_fc_a._modules[str(index)].weight.data = model.classifier._modules[str(index)].weight.data.clone()
              Sherlock_Net.net_po_fc_a._modules[str(index)].bias.data = model.classifier._modules[str(index)].bias.data.clone()
              Sherlock_Net.net_s_fc_a._modules[str(index)].weight.data  = model.classifier._modules[str(index)].weight.data.clone()
              Sherlock_Net.net_s_fc_a._modules[str(index)].bias.data  = model.classifier._modules[str(index)].bias.data.clone()
          else:
              Sherlock_Net.net_po_fc_a._modules[str(index)] = model.classifier._modules[str(index)]
              Sherlock_Net.net_s_fc_a._modules[str(index)]  = model.classifier._modules[str(index)]


    #initializing the last layers
    Sherlock_Net.net_s_fc_b._modules['1'].weight.data.normal_(0, 0.005)
    Sherlock_Net.net_s_fc_b._modules['1'].bias.data.fill_(1)
    #m.bias.data.fill_(0)
    
    Sherlock_Net.net_po_fc_b._modules['1'].weight.data.normal_(0, 0.005)
    Sherlock_Net.net_po_fc_b._modules['1'].bias.data.fill_(1)

    print('Sherlock_Net Initialized with VGG parameters') 
    
    del model
    return Sherlock_Net








#####################################################################################################

#Initialize from VGG checker
def sanity_check(source, copy, starter):
    param_source = list(source.parameters())
    param_copy   = list(copy.parameters())
        
    for i in range(len(param_copy)-1):
        print(param_source[i+starter].data.equal(param_copy[i].data)) 

        