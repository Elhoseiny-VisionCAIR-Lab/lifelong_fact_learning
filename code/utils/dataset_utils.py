#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:42:00 2017

@author: raljundi
"""
#import packages
##################################################################################################################
import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFile
#expert gate rebuttal
import torch.nn as nn
from torch.autograd import Variable



###################################################################################################
'DEFINE YOUR ROOT'
r"""
directories:
- NLP_feat - containing NLP features for test and train
- test - containing test images
- train - containing train images

csv files:
- train_SPO_df.csv: dataframe with train information 
- test_SPO_df.csv:  dataframe with test information

#additional files(to run sanity check):
- Unique_Tuple_maps.txt: dataframe with id and extra info for each Unique fact 
- SPO_w_feat.txt:        dataframe with NLP feat. for each Unique fact.
- SPO_mean_w_feat:       dataframe with mean NLP feat. 

"""
ImageFile.LOAD_TRUNCATED_IMAGES = True
#dataset functions
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10.001)  # pause a bit so that plots are updated

def pil_loader(path, root=None):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    path_png = path.split('.')[0] + '.png'
    # if path == root + 'test/6DS_test_0000003580.jpg':
    #     path = root + 'test/6DS_test_0000003580.png'
    #
    # if path == root + 'test/6DS_test_0000004996.jpg':
    #     path = root + 'test/6DS_test_0000004996.png'
    #
    # if path == root + 'test/6DS_test_0000006301.jpg':
    #     path = root + 'test/6DS_test_0000006301.png'
        
    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    except FileNotFoundError as e:
        with open(path_png, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

def NLP_loader(path):
    NLP_SPO=np.loadtxt(path)
    NLP_SPO=torch.from_numpy(NLP_SPO)
    return NLP_SPO
##################################################################################
##########################################################################################################3
#expert gate rebuttal
def load_eval_dataset(root, test_data_path, batch=1):
    'LOAD YOUR DATASET INFORMATION'
    r"""

    df_train and df_test are dataframe holding:
        - image_links: rel_path to each image
        - NLP_links:   rel_path to each NLP representation of each image
        - SPO: fact representation S:subject, P:Predicate, O:Object
        - id : fact label. (each fact has its own Unique label)
        - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

    """
    # you are going to process one image at the time if you do not state differently
    if isinstance(test_data_path, pd.DataFrame):
        df_test = test_data_path
    else:
        df_test= pd.read_csv(test_data_path)
    
    ##################################################################################################################

    'MAKE YOUR  DATASET'
    # your dataloader will hold
    # images, NLP, ws, wp, wo, labels = data 
    test_dt = Cst_eval_Dataset(df_test['image_links'], df_test['id'], root, image_loader=pil_loader, transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
        ]))

    del  df_test

    # Make your dataset accessible in batches
    dset_loaders = {  'val':torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=False, num_workers=4)}
    return dset_loaders

#make eval Dataset
def make_Cst_eval_dataset( X_image, Unique_fact_id, folder):
    """
    In:
    folder = root_folder
    X_image = your input image
    Unique_fact_id = your facts id , which can be seen as a label
    
    Out:
    facts(image_link, label)
    """
     
    facts = []   
    for i in range(len(X_image)):  
        
        image_path = os.path.join(folder, X_image[i]).replace("'", "")        
        item = [image_path, Unique_fact_id[i]]
        facts.append(item)
    return facts

#you dataset for evaluation will hold just the input (not the target )
class Cst_eval_Dataset(Dataset):
    def __init__(self, X_image, Unique_fact_id, root_dir, image_loader=pil_loader, transform=None):
        r"""
        Build Fact Dataset for evaluation.
        inputs:
            - images path: Images are your Sherlock_Net input
            - NLP_path, Unique_fact_id, w_s, w_p, w_o: These information are your target
            - root_dir
            - image_loader, NLP_loader: These are the method to load your inputs(images) and target(NLP_feat) during training
            - transform: Specify the transformation that you will do on your images
            
        outputs:
            - img: torch.FloatTensor of size 3x224x224  (test_dt["num_image"][0])
            - img_path: relative path to image 
            
        """       
        fact=make_Cst_eval_dataset(X_image, Unique_fact_id, root_dir)
        
        self.fact=fact
        self.root_dir = root_dir
        self.transform = transform
        self.image_loader= image_loader

        
    def __len__(self):
        return len(self.fact)

    def __getitem__(self, idx):
        image_path, label= self.fact[idx]
        img = self.image_loader(image_path, self.root_dir)
        
        if self.transform :
            img = self.transform(img)
        #print(type(image_path))
        return img, image_path


####################################################################################################
#expert gate rebuttal
def make_Cst_dataset_feat_extraction( X_image, folder):
    facts = []   
    for i in range(len(X_image)):        
        image_path = folder + X_image[i]
        facts.append(image_path)
    return facts


class Cst_dataset_feat_extraction(Dataset):
    def __init__(self, X_image,root_dir,image_loader=pil_loader, transform=None):

        fact=make_Cst_dataset_feat_extraction(X_image, root_dir)
        self.fact=fact
        self.root_dir = root_dir
        self.transform = transform
        self.image_loader= image_loader
        
    def __len__(self):
        return len(self.fact)

    def __getitem__(self, idx):
        image_path = self.fact[idx]
        img = self.image_loader(image_path, root=self.root_dir)
        
        if self.transform :
            img = self.transform(img)
        return img, image_path

def load_dataset_feat_extraction(root,train_data_path, test_data_path,batch, shuffle=False):
    df_train = pd.read_csv(train_data_path)
    df_test= pd.read_csv(test_data_path)
    
    ##################################################################################################################

    'MAKE YOUR  DATASET'
    # your dataloader will hold
    # images, NLP, ws, wp, wo, labels = data 
    train_dt = Cst_dataset_feat_extraction(df_train['image_links'],root, image_loader=pil_loader,  transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.CenterCrop(224),                                                                       
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std
                                                                        ]))


    test_dt = Cst_dataset_feat_extraction(df_test['image_links'],root, image_loader=pil_loader,transform = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                        ]))

    del  df_train, df_test

    # Make your dataset accessible in batches
    dset_loaders = {'train': torch.utils.data.DataLoader(train_dt, batch_size=batch, shuffle=shuffle, num_workers=4),
                    'val':torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=shuffle, num_workers=4)}

    return dset_loaders

##################################################################################
###############################################################33
#start expert gate  cvpr rebuttal
def npy_loader(feat_path):
    feat=np.load(feat_path)
    return feat

def make_Cst_autoencoder_dataset( X_feat, feat_root):      
    facts = []  
    for i in range(len(X_feat)):  
        feat_path = feat_root + X_feat[i]
        facts.append(feat_path)
    return facts

class Cst_autoencoder_dataset(Dataset):
    def __init__(self,  X_feat, feat_root, feat_loader = npy_loader):
        
        fact= make_Cst_autoencoder_dataset(X_feat, feat_root)
        self.fact=fact
        self.feat_root = feat_root
        self.feat_loader= feat_loader
        
    def __len__(self):
        return len(self.fact)

    def __getitem__(self, idx):
        feat_path = self.fact[idx]  
        feat=self.feat_loader(feat_path)
        return feat


#end expert gate  cvpr rebuttal
###############################################################
###################################################################################

def make_Cst_dataset( X_image, y_NLP, Unique_fact_id, w_s, w_p, w_o, folder):
    """
    In:
    folder = root_folder
    X_image = your input image
    y_NPL = your target NPL feature vector
    Unique_fact_id = your facts id , which can be seen as a label
    w_s = 1   each fact has the subject
    w_p = 1,0 depends if the fact has (1) or no (0) the predicate
    w_o = 1,0 depends if the fact has (1) or no (0) the object
    
    Out:
    facts(image_link, NLP_link, label, w_s, w_p, w_o)
    """
     
    facts = []   
    for i in range(len(X_image)):  
        
        image_path = folder + 'images/' +  X_image[i]
        NLP_path = folder + 'NLP_feat/' + y_NLP[i]
 
        item = [image_path, NLP_path, w_s[i], w_p[i], w_o[i], Unique_fact_id[i]]
        facts.append(item)
    return facts


class Cst_Dataset(Dataset):
    def __init__(self, X_image, y_NLP, Unique_fact_id, w_s, w_p, w_o, root_dir, image_loader=pil_loader, NLP_feat_loader = NLP_loader, transform=None):
        r"""
        Build Fact Dataset.
        inputs:
            - images path: Images are your Sherlock_Net input
            - NLP_path, Unique_fact_id, w_s, w_p, w_o: These information are your target
            - root_dir
            - image_loader, NLP_loader: These are the method to load your inputs(images) and target(NLP_feat) during training
            - transform: Specify the transformation that you will do on your images
            
        outputs:
            - img: torch.FloatTensor of size 3x224x224  (test_dt["num_image"][0])
            - NLP_SPO: torch.DoubleTensor of size 900   (test_dt["num_image"][1])
            - w_s, w_p, w_o : boolean                   (test_dt["num_image"][2], test_dt["num_image"][3], test_dt["num_image"][4])
            - label: number                             (test_dt["num_image"][5])
            
        """
        
        fact=make_Cst_dataset(X_image, y_NLP, Unique_fact_id, w_s, w_p, w_o, root_dir)
        self.fact=fact
        self.root_dir = root_dir
        self.transform = transform
        self.image_loader= image_loader
        self.NLP_loader= NLP_feat_loader
        
    def __len__(self):
        return len(self.fact)

    def __getitem__(self, idx):
        image_path, NLP_path, w_s, w_p, w_o,label = self.fact[idx]        
        NLP_SPO = self.NLP_loader(NLP_path)
        img = self.image_loader(image_path, root=self.root_dir)
        
        if self.transform :
            img = self.transform(img)
        
        return img, NLP_SPO, w_s, w_p, w_o, label
 
#################################################################################################################
#################################################################################################################
#extract CV features functions

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
    name=image_path.replace(root + 'test/6DS_', 'test/6DS_test_CV_')
    name=name.replace('.jpg', '.txt')
    #save it as txt
         
    np.savetxt(save_dir + name, V) 
    print ('saved:',save_dir + name)
    

#


    
#make eval Dataset
def make_Cst_eval_dataset( X_image, Unique_fact_id, folder):
    """
    In:
    folder = root_folder
    X_image = your input image
    Unique_fact_id = your facts id , which can be seen as a label
    
    Out:
    facts(image_link, label)
    """
     
    facts = []   
    for i in range(len(X_image)):  
        
        image_path = os.path.join(folder, 'images', X_image[i]).replace("'", "")
        item = [image_path, Unique_fact_id[i]]
        facts.append(item)
    return facts



class Cst_Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(img, tuple) :
                img = map(lambda x: t(x), img)
            else:
                img=t(img)   
        return img


#you dataset for evaluation will hold just the input (not the target )
class Cst_eval_Dataset(Dataset):
    def __init__(self, X_image, Unique_fact_id, root_dir, image_loader=pil_loader, transform=None):
        r"""
        Build Fact Dataset for evaluation.
        inputs:
            - images path: Images are your Sherlock_Net input
            - NLP_path, Unique_fact_id, w_s, w_p, w_o: These information are your target
            - root_dir
            - image_loader, NLP_loader: These are the method to load your inputs(images) and target(NLP_feat) during training
            - transform: Specify the transformation that you will do on your images
            
        outputs:
            - img: torch.FloatTensor of size 3x224x224  (test_dt["num_image"][0])
            - img_path: relative path to image 
            
        """       
        fact=make_Cst_eval_dataset(X_image, Unique_fact_id, root_dir)
        
        self.fact=fact
        self.root_dir = root_dir
        self.transform = transform
        self.image_loader= image_loader

        
    def __len__(self):
        return len(self.fact)

    def __getitem__(self, idx):
        image_path, label= self.fact[idx]
        img = self.image_loader(image_path, self.root_dir)
        
        if self.transform :
            img = self.transform(img)
        return img, image_path
 
