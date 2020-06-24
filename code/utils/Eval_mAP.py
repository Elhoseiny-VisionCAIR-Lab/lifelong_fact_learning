#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:37:01 2017

@author: raljundi & fbabilon
"""




import sys
import scipy.io as sio
import pandas as pd
import numpy as np
import pdb
import os
import time
import torch
import collections
import shutil
from torchvision import transforms
from Fact_learning_utils import dataset_utils
from Fact_learning_utils import train_eval_utils
from Fact_learning_utils import sherlock_model_utils 
from Fact_learning_utils import performance_utils
#from  Eval_mAP import *

def fill_default_vals(root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path):
    func_params=[]
    print('-' * 40)
    print('YOUR PARAMETERS')
    print('-' * 40)
    
    #To extract CV features you will need
    if root is None:
        root= '/users/visics/fbabilon/Sherlock/export_Sherlock/'
    print('the root directory is: ', root) 
        
          
    if save_CV_dir  is None:           
        save_CV_dir = 'CV_feat/' + '/test/' 
    print('the exporting directory is: ', save_CV_dir)
      
    if model_to_evaluate_path is None:           
        model_to_evaluate_path = root+'pytorch_models'+'/B1_elastic/' + '/checkpoint.pth.tar'
    print('the model to evaluate is:', model_to_evaluate_path)
        
    if test_data_path is None:
        test_data_path=root + 'data_info/test_SPO_df.cvs'
    print('the test data path is: ',test_data_path)
    
    if target_batch_path is None:
        target_batch_path=root + 'data_splits/B2_test.cvs'
    print('the target batch path is: ',target_batch_path)
    
    if batch is None:
        batch=35
    print('you will load batch of %s '%str(batch))
    
    
    
    #To compute mAP you will need  
    #import your text embedding (NLP feat)
    if NLP_feat_path is None:
        NLP_feat_path= root + '/data_info/TEmbedding.mat'
    print('Your text Embedding is saved in:', NLP_feat_path)
        
    #import info about unique facts    
    if unique_facts_id_path is None:
        unique_facts_id_path= root + 'data_info/Unique_facts_df.cvs'
    print('Your unique facts ids info are saved in:', unique_facts_id_path)
    
    #define distance metric between NLP feat& CV feat
    if score_type is None:
        score_type='euc'
    print('Distance metric is', score_type)
    
    #define distance metric between NLP featu & CV feat
    if ascending_flag is None:
        ascending_flag=0 #this should be 0 in case of euc, 1 in case of cos plain_euc or dot
    print('Your score will be ranked in an ascending order:', score_type)
    
    #define @ in mAP@: numer of image to consider to compute ranking
    if k is None:
        k=100 
    print('You will consider the best %s ranked images:'%str(k))
        
    if dict_pairs_path is None:
        dict_pairs_path = root + '/data_info/dict_pairs.pth.tar'
    print('Dict pairs is saved in:', dict_pairs_path)

    
    print('-' * 40)
    return (root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path)


def extract_feat_mat(root=None, model_to_evaluate_path=None, test_data_path=None,target_batch_path=None, save_CV_dir=None, NLP_feat_path=None, unique_facts_id_path=None, batch=None, score_type=None, ascending_flag=None, k=None, dict_pairs_path=None):
    
    'DEINE YOUR DIRECTORIES'
    root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path = fill_default_vals(root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path)
    df_test= pd.read_csv(test_data_path)
    #make your save directory
    if  os.path.exists(save_CV_dir ):
     
        shutil.rmtree(save_CV_dir)
    if 1:
        print('extracting the features')
        os.makedirs(save_CV_dir)



        'LOAD YOUR DATASET INFORMATION'
        r"""
        df_test are dataframe holding:
            - image_links: rel_path to each image
            - NLP_links:   rel_path to each NLP representation of each image
            - SPO: fact representation S:subject, P:Predicate, O:Object
            - id : fact label. (each fact has its own Unique label)
            - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

        """


        ##########################################################################################################
        'MAKE YOUR  DATASET'
        # your dataloader will hold
        # images, labels = data 
        test_dt = dataset_utils.Cst_eval_Dataset(df_test['image_links'], df_test['id'], root, image_loader=dataset_utils.pil_loader, transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
        ]))
        # Make your dataset accessible in batches
        dset_loaders = {'val': torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=False, num_workers=4)}

        use_gpu = torch.cuda.is_available()
        print('Dataset Loaded. your batch size is %s'%batch)
        print('Use GPU%s '%use_gpu)  

        #load the model 
        print('Loading model to test') 
        checkpoint=torch.load(model_to_evaluate_path)
        try:
            Sherlock_Net=checkpoint['model']
        except:
            Sherlock_Net=checkpoint
        if use_gpu:
            try:
            #pdb.set_trace()
                Sherlock_Net=Sherlock_Net.module
            except AttributeError:
                print('No module to remove')
            Sherlock_Net=Sherlock_Net.cuda()


                ###########################################################################################################
        'EXTRACT CV FEATURES'
                #you will pass each image through the net and save the net output in txt file.
                #check
                #pdb.set_trace()
        train_eval_utils.eval_fact_model_mat(Sherlock_Net, dset_loaders, root, save_CV_dir) 

        

##########################################################################################################


def extract_feat_mat(root=None, model_to_evaluate_path=None, test_data_path=None, target_batch_path=None,
                     save_CV_dir=None, NLP_feat_path=None, unique_facts_id_path=None, batch=None, score_type=None,
                     ascending_flag=None, k=None, dict_pairs_path=None):
    'DEINE YOUR DIRECTORIES'
    root, model_to_evaluate_path, test_data_path, target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path = fill_default_vals(
        root, model_to_evaluate_path, test_data_path, target_batch_path, save_CV_dir, NLP_feat_path,
        unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path)
    df_test = pd.read_csv(test_data_path)
    # make your save directory
    if os.path.exists(save_CV_dir):
        shutil.rmtree(save_CV_dir)
    if 1:
        print('extracting the features')
        os.makedirs(save_CV_dir)

        'LOAD YOUR DATASET INFORMATION'
        r"""
        df_test are dataframe holding:
            - image_links: rel_path to each image
            - NLP_links:   rel_path to each NLP representation of each image
            - SPO: fact representation S:subject, P:Predicate, O:Object
            - id : fact label. (each fact has its own Unique label)
            - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

        """

        ##########################################################################################################
        'MAKE YOUR  DATASET'
        # your dataloader will hold
        # images, labels = data
        test_dt = dataset_utils.Cst_eval_Dataset(df_test['image_links'], df_test['id'], root,
                                                 image_loader=dataset_utils.pil_loader, transform=transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # per-channel mean/std
            ]))
        # Make your dataset accessible in batches
        dset_loaders = {'val': torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=False, num_workers=4)}

        use_gpu = torch.cuda.is_available()
        print('Dataset Loaded. your batch size is %s' % batch)
        print('Use GPU%s ' % use_gpu)

        # load the model
        print('Loading model to test')
        checkpoint = torch.load(model_to_evaluate_path)
        try:
            Sherlock_Net = checkpoint['model']
        except:
            Sherlock_Net = checkpoint
        if use_gpu:
            try:
                # pdb.set_trace()
                Sherlock_Net = Sherlock_Net.module
            except AttributeError:
                print('No module to remove')
            Sherlock_Net = Sherlock_Net.cuda()

            ###########################################################################################################
        'EXTRACT CV FEATURES'
        # you will pass each image through the net and save the net output in txt file.
        # check
        # pdb.set_trace()
        train_eval_utils.eval_fact_model_mat(Sherlock_Net, dset_loaders, root, save_CV_dir)

    ##########################################################################################################


def extract_feat_mat_oneach(root=None, model_to_evaluate_path=None, test_data_path=None, target_batch_path=None,
                     save_CV_dir=None, NLP_feat_path=None, unique_facts_id_path=None, batch=None, score_type=None,
                     ascending_flag=None, k=None, dict_pairs_path=None):
    'DEINE YOUR DIRECTORIES'
    root, model_to_evaluate_path, test_data_path, target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path = fill_default_vals(
        root, model_to_evaluate_path, test_data_path, target_batch_path, save_CV_dir, NLP_feat_path,
        unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path)
    df_test = pd.read_csv(target_batch_path)
    # make your save directory
    if os.path.exists(save_CV_dir):
        shutil.rmtree(save_CV_dir)
    if 1:
        print('extracting the features')
        os.makedirs(save_CV_dir)

        'LOAD YOUR DATASET INFORMATION'
        r"""
        df_test are dataframe holding:
            - image_links: rel_path to each image
            - NLP_links:   rel_path to each NLP representation of each image
            - SPO: fact representation S:subject, P:Predicate, O:Object
            - id : fact label. (each fact has its own Unique label)
            - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

        """

        ##########################################################################################################
        'MAKE YOUR  DATASET'
        # your dataloader will hold
        # images, labels = data
        test_dt = dataset_utils.Cst_eval_Dataset(df_test['image_links'], df_test['id'], root,
                                                 image_loader=dataset_utils.pil_loader, transform=transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # per-channel mean/std
            ]))
        # Make your dataset accessible in batches
        dset_loaders = {'val': torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=False, num_workers=4)}

        use_gpu = torch.cuda.is_available()
        print('Dataset Loaded. your batch size is %s' % batch)
        print('Use GPU%s ' % use_gpu)

        # load the model
        print('Loading model to test')
        checkpoint = torch.load(model_to_evaluate_path)
        try:
            Sherlock_Net = checkpoint['model']
        except:
            Sherlock_Net = checkpoint
        if use_gpu:
            try:
                # pdb.set_trace()
                Sherlock_Net = Sherlock_Net.module
            except AttributeError:
                print('No module to remove')
            Sherlock_Net = Sherlock_Net.cuda()

            ###########################################################################################################
        'EXTRACT CV FEATURES'
        # you will pass each image through the net and save the net output in txt file.
        # check
        # pdb.set_trace()
        train_eval_utils.eval_fact_model_mat(Sherlock_Net, dset_loaders, root, save_CV_dir)

    ##########################################################################################################


def eval_map(root=None, model_to_evaluate_path=None, test_data_path=None,target_batch_path=None, save_CV_dir=None, NLP_feat_path=None, unique_facts_id_path=None, batch=None, score_type=None, ascending_flag=None, k=None, dict_pairs_path=None):
    
    'DEINE YOUR DIRECTORIES'
    root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path = fill_default_vals(root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path)
    df_test= pd.read_csv(test_data_path)
    #make your save directory
    if not os.path.exists(save_CV_dir ):
        print('extracting the features')
        os.makedirs(save_CV_dir)
        

        
        'LOAD YOUR DATASET INFORMATION'
        r"""
        df_test are dataframe holding:
            - image_links: rel_path to each image
            - NLP_links:   rel_path to each NLP representation of each image
            - SPO: fact representation S:subject, P:Predicate, O:Object
            - id : fact label. (each fact has its own Unique label)
            - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

        """
        

        ##################################################################################################################
        'MAKE YOUR  DATASET'
        # your dataloader will hold
        # images, labels = data 
        test_dt = dataset_utils.Cst_eval_Dataset(df_test['image_links'], df_test['id'], root, image_loader=dataset_utils.pil_loader, transform = transforms.Compose([
                                                                                                                             transforms.Scale(256),
                                                                                                                             transforms.CenterCrop(224),
                                                                                                                             transforms.ToTensor(),
                                                                                                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                                                                             ]))
        # Make your dataset accessible in batches
        dset_loaders = {'val': torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=False, num_workers=4)}

        use_gpu = torch.cuda.is_available()
        print('Dataset Loaded. your batch size is %s'%batch)
        print('Use GPU%s '%use_gpu)  

        #load the model 
        print('Loading model to test') 
        checkpoint=torch.load(model_to_evaluate_path)
        try:
            Sherlock_Net=checkpoint['model']
        except TypeError:
            Sherlock_Net=checkpoint
        if use_gpu:
            try:
                Sherlock_Net=Sherlock_Net.module
            except AttributeError:
                print('No module to remove')
            Sherlock_Net=Sherlock_Net.cuda()
           
                
            

        ##################################################################################################################
        'EXTRACT CV FEATURES'
        #you will pass each image through the net and save the net output in txt file.
        #check
        #pdb.set_trace()
        train_eval_utils.eval_fact_model(Sherlock_Net, dset_loaders, root, save_CV_dir) 
    
    'MAKE RESULT DF'
    #save a dataframe with your results
    CV_feat = os.listdir(save_CV_dir )
    #CV_feat = list(map(lambda i: i.replace('6DS_','test/6DS_'), CV_feat)) 
    CV_feat.sort()
    feat_df=df_test
    feat_df['CV_feat'] = CV_feat
    
    #df1=pd.DataFrame({'CV_feat': CV_feat})
    #feat_df=pd.concat([df_test, df1], axis=1)
    #feat_df.to_csv(save_CV_dir + 'CV_results.cvs', index=False)
    
    #################################################################################################################
    'COMPUTE THE MEAN AVERAGE PRECISION '
   
    #load the Unique facts dataframe
    facts_df = pd.read_csv(unique_facts_id_path) 
    
    #select from the list of unique ids present in your test images
    
    Batch_facts=pd.read_csv(target_batch_path)
    facts_to_evaluate=(sorted(list(set(Batch_facts['id']))))
    #facts_to_evaluate=list(set(feat_df['id']))
    
    print('importing CV features')
    #pdb.set_trace()
    CV_feat = [ performance_utils.feat_loader(save_CV_dir + i) for i in feat_df['CV_feat']]
    CV_feat = np.asarray(CV_feat)
    
    #get original Text Embedding feat images*900
    TEmbedding=sio.loadmat(NLP_feat_path) 
    TEmbedding=TEmbedding['TEmbedding']

    #compute  mAp_k (over k images)     
    py_mAp_k=[] 

    #for each fact, get a mAP100 score
    for fact_id in facts_to_evaluate:
        #pdb.set_trace()
        mAp_k=performance_utils.ap_k(fact_id, k, score_type, CV_feat, feat_df, facts_df, root, ascending_flag )
        # print (fact_id, 'mAP100:%s'%str(mAp_k))
        py_mAp_k.append(mAp_k)

    #then print as evaluation metric the average mAP100    
    result=np.mean(np.asarray(py_mAp_k))
    print('Averaged mAP%s:'%str(k), result)
    return result

####NOT HARSH GOES######################################


def eval_map_not_harsh_onall(root=None, model_to_evaluate_path=None, test_data_path=None,target_batch_path=None, save_CV_dir=None, NLP_feat_path=None, unique_facts_id_path=None, batch=None, score_type=None, ascending_flag=None, k=None, dict_pairs_path=None):
    
    'DEINE YOUR DIRECTORIES'
    root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path = fill_default_vals(root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path)
    df_test= pd.read_csv(test_data_path)
    #make your save directory
    if not os.path.exists(save_CV_dir ):
        print('extracting the features')
        os.makedirs(save_CV_dir)
        

        
        'LOAD YOUR DATASET INFORMATION'
        r"""
        df_test are dataframe holding:
            - image_links: rel_path to each image
            - NLP_links:   rel_path to each NLP representation of each image
            - SPO: fact representation S:subject, P:Predicate, O:Object
            - id : fact label. (each fact has its own Unique label)
            - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

        """
        

        ###########################################################################################################
        'MAKE YOUR  DATASET'
        # your dataloader will hold
        # images, labels = data 
        test_dt = dataset_utils.Cst_eval_Dataset(df_test['image_links'], df_test['id'], root, image_loader=dataset_utils.pil_loader, transform = transforms.Compose([
                                                                                                                             transforms.Scale(256),
                                                                                                                             transforms.CenterCrop(224),
                                                                                                                             transforms.ToTensor(),
                                                                                                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                                                                             ]))
        # Make your dataset accessible in batches
        dset_loaders = {'val': torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=False, num_workers=4)}

        use_gpu = torch.cuda.is_available()
        print('Dataset Loaded. your batch size is %s'%batch)
        print('Use GPU%s '%use_gpu)  

        #load the model 
        print('Loading model to test') 
        checkpoint=torch.load(model_to_evaluate_path)
        try:
            Sherlock_Net=checkpoint['model']
        except TypeError:
            Sherlock_Net=checkpoint
        if use_gpu:
            Sherlock_Net=Sherlock_Net.cuda()


        ###########################################################################################################
        'EXTRACT CV FEATURES'
        #you will pass each image through the net and save the net output in txt file.
        #check
        #pdb.set_trace()
        train_eval_utils.eval_fact_model(Sherlock_Net, dset_loaders, root, save_CV_dir) 
    
     
    #################################################################################################################
    'COMPUTE THE MEAN AVERAGE PRECISION NOT HARSH '
    
    #load the Unique facts dataframe
    facts_df = pd.read_csv(unique_facts_id_path) 
    
    #select from the list of unique ids present in your test images
    
    Batch_df=pd.read_csv(target_batch_path)
    
    facts_to_evaluate=(sorted(list(set(Batch_df['id']))))
    facts_to_evaluate_df=facts_df[facts_df['id'].isin(facts_to_evaluate)]
    #facts_to_evaluate=list(set(feat_df['id']))
    
    'MAKE RESULT DF'
    #and you can add here the CV feat extracted by the model
    # ON ALL, in this case the CV_feat should be 14025 all the time
    feat_df = df_test
    feat_df = feat_df.reset_index()
    
    # And you can add here the CV feat extracted by the model
    CV_feat_list=[]
    for i in range(len(feat_df)):
        CV_feat_list.append(feat_df['image_links'].values[i].split('/')[1].replace('.jpg', '.txt'))

    feat_df['CV_feat'] = CV_feat_list
    
    #df1=pd.DataFrame({'CV_feat': CV_feat})
    #feat_df=pd.concat([df_test, df1], axis=1)
    #feat_df.to_csv(save_CV_dir + 'CV_results.cvs', index=False)
  
    
    #merge duplicate
    #Find and Save Dataset Pairs    
    if os.path.exists(dict_pairs_path):
        print('import facts duplicate dictionary')
        pairs=torch.load(dict_pairs_path)

    else:
        print('built facts duplicate dictionary')
        pairs=performance_utils.find_dataset_pairs(facts, root)
        torch.save(pairs, dict_pairs_path)

    print('Merging duplicate')
    performance_utils.merge_duplicate(facts_to_evaluate_df, feat_df, pairs)
    facts_to_evaluate=(sorted(list(set(facts_to_evaluate_df['id']))))
    #recount the facts to evaluate
    
    
    
    #pdb.set_trace()
    CV_feat = [ performance_utils.feat_loader(save_CV_dir + i) for i in feat_df['CV_feat']]
    CV_feat = np.asarray(CV_feat)
    print('importing CV features on all,the shape should be TEST_df_size, 900', CV_feat.shape)
    
    #get original Text Embedding feat images*900
    TEmbedding=sio.loadmat(NLP_feat_path) 
    TEmbedding=TEmbedding['TEmbedding']
    
    

    #compute  mAp_k (over k images)     
    py_mAp_k=[] 

    #for each fact, get a mAP100 score
    for fact_id in facts_to_evaluate:
        #pdb.set_trace()
        mAp_k=performance_utils.ap_k(fact_id, k, score_type, CV_feat, feat_df, facts_df, root, ascending_flag )
        # print (fact_id, 'mAP100:%s'%str(mAp_k))
        py_mAp_k.append(mAp_k)

    #then print as evaluation metric the average mAP100    
    result=np.mean(np.asarray(py_mAp_k))
    print('Averaged not harsh mAP%s:'%str(k), result)
    return result
########################################################################################################################
def eval_map_not_harsh_oneach(root=None, model_to_evaluate_path=None, test_data_path=None,target_batch_path=None, save_CV_dir=None, NLP_feat_path=None, unique_facts_id_path=None, batch=None, score_type=None, ascending_flag=None, k=None, dict_pairs_path=None):
    
    'DEINE YOUR DIRECTORIES'
    root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path = fill_default_vals(root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path)
    df_test= pd.read_csv(test_data_path)
    #make your save directory
    if not os.path.exists(save_CV_dir ):
        print('extracting the features')
        os.makedirs(save_CV_dir)
        

        
        'LOAD YOUR DATASET INFORMATION'
        r"""
        df_test are dataframe holding:
            - image_links: rel_path to each image
            - NLP_links:   rel_path to each NLP representation of each image
            - SPO: fact representation S:subject, P:Predicate, O:Object
            - id : fact label. (each fact has its own Unique label)
            - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

        """
        

        ###########################################################################################################
        'MAKE YOUR  DATASET'
        # your dataloader will hold
        # images, labels = data 
        test_dt = dataset_utils.Cst_eval_Dataset(df_test['image_links'], df_test['id'], root, image_loader=dataset_utils.pil_loader, transform = transforms.Compose([
                                                                                                                             transforms.Scale(256),
                                                                                                                             transforms.CenterCrop(224),
                                                                                                                             transforms.ToTensor(),
                                                                                                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                                                                             ]))
        # Make your dataset accessible in batches
        dset_loaders = {'val': torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=False, num_workers=4)}

        use_gpu = torch.cuda.is_available()
        print('Dataset Loaded. your batch size is %s'%batch)
        print('Use GPU%s '%use_gpu)  

        #load the model 
        print('Loading model to test') 
        checkpoint=torch.load(model_to_evaluate_path)
        try:
            Sherlock_Net=checkpoint['model']
        except TypeError:
            Sherlock_Net=checkpoint
        if use_gpu:
            Sherlock_Net=Sherlock_Net.cuda()


        ###########################################################################################################
        'EXTRACT CV FEATURES'
        #you will pass each image through the net and save the net output in txt file.
        #check
        #pdb.set_trace()
        train_eval_utils.eval_fact_model(Sherlock_Net, dset_loaders, root, save_CV_dir) 
    
     
    #################################################################################################################
    'COMPUTE THE MEAN AVERAGE PRECISION NOT HARSH '
    
    #load the Unique facts dataframe
    facts_df = pd.read_csv(unique_facts_id_path) 
    
    #select from the list of unique ids present in your test images
    
    Batch_df=pd.read_csv(target_batch_path)
    
    facts_to_evaluate=(sorted(list(set(Batch_df['id']))))
    facts_to_evaluate_df=facts_df[facts_df['id'].isin(facts_to_evaluate)]
    #facts_to_evaluate=list(set(feat_df['id']))
    
    'MAKE RESULT DF'
    #and you can add here the CV feat extracted by the model
    # ON EACH, in this case the CV_feat should be big as the batch len.
    feat_df = Batch_df
    feat_df = feat_df.reset_index()

    # And you can add here the CV feat extracted by the model
    CV_feat_list=[]
    for i in range(len(feat_df)):
        CV_feat_list.append(feat_df['image_links'].values[i].split('/')[1].replace('.jpg', '.txt'))

    feat_df['CV_feat'] = CV_feat_list
    
    #df1=pd.DataFrame({'CV_feat': CV_feat})
    #feat_df=pd.concat([df_test, df1], axis=1)
    #feat_df.to_csv(save_CV_dir + 'CV_results.cvs', index=False)
  
    
    #merge duplicate
    #Find and Save Dataset Pairs    
    if os.path.exists(dict_pairs_path):
        print('import facts duplicate dictionary')
        pairs=torch.load(dict_pairs_path)

    else:
        print('built facts duplicate dictionary')
        pairs=performance_utils.find_dataset_pairs(facts, root)
        torch.save(pairs, dict_pairs_path)

    print('Merging duplicate')
    performance_utils.merge_duplicate(facts_to_evaluate_df, feat_df, pairs)
    facts_to_evaluate=(sorted(list(set(facts_to_evaluate_df['id']))))
    #recount the facts to evaluate
    
    
    
    #pdb.set_trace()
    CV_feat = [ performance_utils.feat_loader(save_CV_dir + i) for i in feat_df['CV_feat']]
    CV_feat = np.asarray(CV_feat)
    print('importing CV features on each, the shape should be BATCH_size, 900', CV_feat.shape)
    
    #get original Text Embedding feat images*900
    TEmbedding=sio.loadmat(NLP_feat_path) 
    TEmbedding=TEmbedding['TEmbedding']
    
    

    #compute  mAp_k (over k images)     
    py_mAp_k=[] 

    #for each fact, get a mAP100 score
    for fact_id in facts_to_evaluate:
        #pdb.set_trace()
        mAp_k=performance_utils.ap_k(fact_id, k, score_type, CV_feat, feat_df, facts_df, root, ascending_flag )
        # print (fact_id, 'mAP100:%s'%str(mAp_k))
        py_mAp_k.append(mAp_k)

    #then print as evaluation metric the average mAP100    
    result=np.mean(np.asarray(py_mAp_k))
    print('Averaged not harsh mAP%s:'%str(k), result)
    return result

########################################################################################################################
def i2f_topk_not_harsh(root=None, model_to_evaluate_path=None, test_data_path=None,target_batch_path=None, save_CV_dir=None, NLP_feat_path=None, unique_facts_id_path=None, batch=None, score_type=None, ascending_flag=None, k=None, dict_pairs_path=None):
    
    'DEINE YOUR DIRECTORIES'
    root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path = fill_default_vals(root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path)
    df_test= pd.read_csv(test_data_path)
    #make your save directory
    if not os.path.exists(save_CV_dir ):
        print('extracting the features')
        os.makedirs(save_CV_dir)
        

        
        'LOAD YOUR DATASET INFORMATION'
        r"""
        df_test are dataframe holding:
            - image_links: rel_path to each image
            - NLP_links:   rel_path to each NLP representation of each image
            - SPO: fact representation S:subject, P:Predicate, O:Object
            - id : fact label. (each fact has its own Unique label)
            - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

        """
        

        ###########################################################################################################
        'MAKE YOUR  DATASET'
        # your dataloader will hold
        # images, labels = data 
        test_dt = dataset_utils.Cst_eval_Dataset(df_test['image_links'], df_test['id'], root, image_loader=dataset_utils.pil_loader, transform = transforms.Compose([
                                                                                                                             transforms.Scale(256),
                                                                                                                             transforms.CenterCrop(224),
                                                                                                                             transforms.ToTensor(),
                                                                                                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                                                                             ]))
        # Make your dataset accessible in batches
        dset_loaders = {'val': torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=False, num_workers=4)}

        use_gpu = torch.cuda.is_available()
        print('Dataset Loaded. your batch size is %s'%batch)
        print('Use GPU%s '%use_gpu)  

        #load the model 
        print('Loading model to test') 
        checkpoint=torch.load(model_to_evaluate_path)
        try:
            Sherlock_Net=checkpoint['model']
        except TypeError:
            Sherlock_Net=checkpoint
        if use_gpu:
            Sherlock_Net=Sherlock_Net.cuda()


        ###########################################################################################################
        'EXTRACT CV FEATURES'
        #you will pass each image through the net and save the net output in txt file.
        #check
        #pdb.set_trace()
        train_eval_utils.eval_fact_model(Sherlock_Net, dset_loaders, root, save_CV_dir) 
         
    
    #################################################################################################################
    'COMPUTE THE TOP K NOT HARSH '
    
    #load the Unique facts dataframe
    facts_df = pd.read_csv(unique_facts_id_path) 
    
    #select from the list of unique ids present in your test images  
    Batch_df = pd.read_csv(target_batch_path)
    facts_to_evaluate = (sorted(list(set(Batch_df['id']))))
    facts_to_evaluate_df = facts_df[facts_df['id'].isin(facts_to_evaluate)]
    #facts_to_evaluate=list(set(feat_df['id']))
      
    #and you can add here the CV feat extracted by the model
    # In this case your feat_df IS BATCH TARGET DF
    feat_df = Batch_df
    feat_df = feat_df.reset_index()
    
    # And you can add here the CV feat extracted by the model
    CV_feat_list=[]
    for i in range(len(feat_df)):
        CV_feat_list.append(feat_df['image_links'].values[i].split('/')[1].replace('.jpg', '.txt'))

    feat_df['CV_feat'] = CV_feat_list
    
    #df1=pd.DataFrame({'CV_feat': CV_feat})
    #feat_df=pd.concat([df_test, df1], axis=1)
    #feat_df.to_csv(save_CV_dir + 'CV_results.cvs', index=False)
    
    
    #merge duplicate
    #Find and Save Dataset Pairs    
    if os.path.exists(dict_pairs_path):
        print('import facts duplicate dictionary')
        pairs = torch.load(dict_pairs_path)

    else:
        print('built facts duplicate dictionary')
        pairs = performance_utils.find_dataset_pairs(facts, root)
        torch.save(pairs, dict_pairs_path)

    print('Merging duplicate')
    performance_utils.merge_duplicate(facts_to_evaluate_df, feat_df, pairs)
    facts_to_evaluate = (sorted(list(set(facts_to_evaluate_df['id']))))
    #recount the facts to evaluate
    
    #Remove the duplicate rows in facts_df 
    facts_to_evaluate_df = facts_to_evaluate_df.drop_duplicates(subset='id')
    facts_to_evaluate_df = facts_to_evaluate_df.reset_index(drop=True)
    facts_to_evaluate = (sorted(list(set(facts_to_evaluate_df['id']))))
    
    print('')
    print('After merging duplicate you have %s facts to evaluate.'%str(len(facts_to_evaluate)), 'Your test image without duplicate are',len(feat_df))
    
    #build you NLP_feats matrix with the facts needed
    flag=False
    for index, row in facts_to_evaluate_df.iterrows():
        feat=np.loadtxt(root + 'NLP_feat/unique_facts/'+ row['NLP_links'])
        if flag :
            matrix= np.vstack((matrix,feat))

        else:
            matrix=feat
            flag=True
    print('NLP matrix built', matrix.shape)


    #get one points
    print('Evaluating Top %s '%str(k))
    points=0
    for image in range(len(feat_df)):
        image_CV_path = save_CV_dir + list(feat_df['CV_feat'])[image]
        image_id = list(feat_df['id'])[image]
        #print(image_CV_path, image_id)

        performance_utils.feat_loader(image_CV_path)
        scores = performance_utils.i2f_scorer(image_CV_path, score_type, matrix, facts_to_evaluate_df)
        
        
        results = pd.concat([facts_to_evaluate_df, scores], axis=1)
        #sort df according to scores 
        sorted_results=results.sort_values(by='%s_score'%score_type, ascending=ascending_flag)
        
        #print(k, image_id, feat_df.iloc[image, :])
        if image_id in sorted_results[0:k]['id'].values:
            points +=1
            #print('image %s'%str(image), 'facts id: %s'%str(image_id), 'Made it !')

    topk = points / float(len(feat_df))
    print('Top%s not harsh over your batch'%str(k), topk)
    return topk

########################################################################################################################

def extract_features(root=None, model_to_evaluate_path=None, test_data_path=None,target_batch_path=None, save_CV_dir=None, NLP_feat_path=None, unique_facts_id_path=None, batch=None, score_type=None, ascending_flag=None, k=None,module_name='S'):
    
    'DEINE YOUR DIRECTORIES'
    root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k,dict_pairs_path = fill_default_vals(root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k,dict_pairs_path=None)
    df_test= pd.read_csv(test_data_path)
    #make your save directory
    if not os.path.exists(save_CV_dir ):
        print('extracting the features')
        os.makedirs(save_CV_dir)
        

        
        'LOAD YOUR DATASET INFORMATION'
        r"""
        df_test are dataframe holding:
            - image_links: rel_path to each image
            - NLP_links:   rel_path to each NLP representation of each image
            - SPO: fact representation S:subject, P:Predicate, O:Object
            - id : fact label. (each fact has its own Unique label)
            - w_s, w_p, w_o: boolean. Indicate if the fact representated in the image has a Subject(w_s), Predicate(w_p), Object(w_o)"

        """
        

        ##################################################################################################################
        'MAKE YOUR  DATASET'
        # your dataloader will hold
        # images, labels = data 
        test_dt = dataset_utils.Cst_eval_Dataset(df_test['image_links'], df_test['id'], root, image_loader=dataset_utils.pil_loader, transform = transforms.Compose([
                                                                                                                             transforms.Scale(256),
                                                                                                                             transforms.CenterCrop(224),
                                                                                                                             transforms.ToTensor(),
                                                                                                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #per-channel mean/std 
                                                                                                                             ]))
        # Make your dataset accessible in batches
        dset_loaders = {'val': torch.utils.data.DataLoader(test_dt, batch_size=batch, shuffle=False, num_workers=4)}

        use_gpu = torch.cuda.is_available()
        print('Dataset Loaded. your batch size is %s'%batch)
        print('Use GPU%s '%use_gpu)  

        #load the model 
        print('Loading model to test') 
        checkpoint=torch.load(model_to_evaluate_path)
        try:
            Sherlock_Net=checkpoint['model']
        except TypeError:
            Sherlock_Net=checkpoint
        if use_gpu:
            Sherlock_Net=Sherlock_Net.cuda()


        ##################################################################################################################
        'EXTRACT CV FEATURES'
        #you will pass each image through the net and save the net output in txt file.
        #check
        #pdb.set_trace()
        if module_name=='S':
            train_eval_utils.extract_S_features(Sherlock_Net, dset_loaders, root, save_CV_dir) 
        else:
            train_eval_utils.extract_PO_features(Sherlock_Net, dset_loaders, root, save_CV_dir) 
        

        'MAKE RESULT DF'
        #save a dataframe with your results
        CV_feat = os.listdir(save_CV_dir )
        #CV_feat = list(map(lambda i: i.replace('6DS_','test/6DS_'), CV_feat)) 
        CV_feat.sort()
        feat_df=df_test
        feat_df['CV_feat'] = CV_feat
        feat_df.to_csv(save_CV_dir + 'CV_results.cvs', index=False)

def test_4tasks_model_onal(root,model_to_evaluate_path,output_results_path,save_CV_dir, batch):
    
    
    target_batch_path=root+'data_splits/4tasks/B1_test.cvs'

    ###########################################################################################################################
    results={}
    #clean the save feature dirs
    if  os.path.exists(save_CV_dir):
        shutil.rmtree(save_CV_dir)
        
    if  not os.path.exists('results'):
        os.mkdir('results')
        
    res_b1=eval_map(root=root, target_batch_path=target_batch_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir, batch=batch)
    results['res_b1']=res_b1

    ###########################################################################################################################
    #!rm -r /esat/jade/raljundi/B2_obj_res2/
    target_batch_path=root+'data_splits/4tasks/B2_test.cvs'

    #save_CV_dir='/esat/jade/raljundi/B2_obj_res2/'

    res_b2=eval_map(root=root, target_batch_path=target_batch_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir, batch=batch)
    results['res_b2']=res_b2

    #!rm -r /esat/jade/raljundi/B2_obj_res3/
    target_batch_path=root+'data_splits/4tasks/B3_test.cvs'

    #save_CV_dir='/esat/jade/raljundi/B2_obj_res3/'

    res_b3=eval_map(root=root, target_batch_path=target_batch_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir, batch=batch)
    results['res_b3']=res_b3

    #!rm -r /esat/jade/raljundi/B2_obj_res4/
    target_batch_path=root+'data_splits/4tasks/B4_test.cvs'

    #save_CV_dir='/esat/jade/raljundi/B2_obj_res4/'

    res_b4=eval_map(root=root, target_batch_path=target_batch_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir, batch=batch)
    results['res_b4']=res_b4

    target_batch_path=root + 'data_info/test_SPO_df.cvs'
    #save_CV_dir='/esat/jade/raljundi/test_all/'
    res_all=eval_map(root=root, model_to_evaluate_path=model_to_evaluate_path, target_batch_path=target_batch_path, save_CV_dir=save_CV_dir, batch=batch)
    results['res_all']=res_all
    torch.save(results,'./results/'+output_results_path)
    print(results)
    return results

def test_4tasks_disjoint_model_onall(root,model_to_evaluate_path,output_results_path,save_CV_dir):
    
    
    target_batch_path=root+'data_splits/4tasks_disjoint/B1_test.cvs'

    ###########################################################################################################################
    results={}
    #clean the save feature dirs
    if  os.path.exists(save_CV_dir):
        shutil.rmtree(save_CV_dir)
        
    if  not os.path.exists('results'):
        os.mkdir('results')
        
    res_b1=eval_map(root=root,target_batch_path=target_batch_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_b1']=res_b1

    ###########################################################################################################################
    #!rm -r /esat/jade/raljundi/B2_obj_res2/
    target_batch_path=root+'data_splits/4tasks_disjoint/B2_test.cvs'

    #save_CV_dir='/esat/jade/raljundi/B2_obj_res2/'

    res_b2=eval_map(root=root,target_batch_path=target_batch_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_b2']=res_b2

    #!rm -r /esat/jade/raljundi/B2_obj_res3/
    target_batch_path=root+'data_splits/4tasks_disjoint/B3_test.cvs'

    #save_CV_dir='/esat/jade/raljundi/B2_obj_res3/'

    res_b3=eval_map(root=root,target_batch_path=target_batch_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_b3']=res_b3

    #!rm -r /esat/jade/raljundi/B2_obj_res4/
    target_batch_path=root+'data_splits/4tasks_disjoint/B4_test.cvs'

    #save_CV_dir='/esat/jade/raljundi/B2_obj_res4/'

    res_b4=eval_map(root=root,target_batch_path=target_batch_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_b4']=res_b4

    target_batch_path=root + 'data_info/test_SPO_df.cvs'
    #if os.path.file.exists(target_batch_path):
    #save_CV_dir='/esat/jade/raljundi/test_all/'
    res_all=eval_map(root=root,model_to_evaluate_path=model_to_evaluate_path, target_batch_path=target_batch_path, save_CV_dir=save_CV_dir)
    results['res_all']=res_all
    torch.save(results,'./results/'+output_results_path)
    print(results)
    return results

def test_4tasks_disjoint_model_oneach(root,model_to_evaluate_path,output_results_path,save_CV_dir):
    
   
    target_batch_path=root+'data_splits/4tasks_disjoint/B1_test.cvs'
    test_data_path=target_batch_path
    ###########################################################################################################################
    results={}
    #clean the save feature dirs
    if  os.path.exists(save_CV_dir):
        shutil.rmtree(save_CV_dir)
        
    if  not os.path.exists('results'):
        os.mkdir('results')
        
    res_b1=eval_map(root=root,test_data_path=test_data_path,target_batch_path=target_batch_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    if  os.path.exists(save_CV_dir):
        shutil.rmtree(save_CV_dir)            
    results['res_b1']=res_b1

    ###########################################################################################################################
    #!rm -r /esat/jade/raljundi/B2_obj_res2/
    target_batch_path=root+'data_splits/4tasks_disjoint/B2_test.cvs'
    test_data_path=target_batch_path
    #save_CV_dir='/esat/jade/raljundi/B2_obj_res2/'

    res_b2=eval_map(root=root,test_data_path=test_data_path,target_batch_path=target_batch_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_b2']=res_b2

    #!rm -r /esat/jade/raljundi/B2_obj_res3/
    target_batch_path=root+'data_splits/4tasks_disjoint/B3_test.cvs'
    test_data_path=target_batch_path
    #save_CV_dir='/esat/jade/raljundi/B2_obj_res3/'
    if  os.path.exists(save_CV_dir):
        shutil.rmtree(save_CV_dir)
    res_b3=eval_map(root=root,test_data_path=test_data_path,target_batch_path=target_batch_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_b3']=res_b3

    #!rm -r /esat/jade/raljundi/B2_obj_res4/
    target_batch_path=root+'data_splits/4tasks_disjoint/B4_test.cvs'
    test_data_path=target_batch_path
    #save_CV_dir='/esat/jade/raljundi/B2_obj_res4/'
    if  os.path.exists(save_CV_dir):
        shutil.rmtree(save_CV_dir)
    res_b4=eval_map(root=root,test_data_path=test_data_path,target_batch_path=target_batch_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_b4']=res_b4

   
    torch.save(results,'./results/'+output_results_path)
    print(results)
    return results

####################################################################################################
def eval_two_Tasks_not_duplicate_i2f_and_f2i(name, model_to_evaluate_path, root, test_data_path, save_CV_dir, k): #print(model_to_evaluate_path, name)
    r"""
    Computes the remaining evaluation metric for 2tasks jobs
    """
    
    if os.path.exists(save_CV_dir):
        shutil.rmtree(save_CV_dir)
    
    #eval_map_not_harsh(root=root,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    
    save_res= root + '/code/results/%s'%str(name)

    results={}
    
    print(' %s ALL'%str(name))
    target_data_path = root+ 'data_info/test_SPO_df.cvs'    
    res_all=eval_map(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_all_harsh'] = res_all
    res_all_nh=eval_map_not_harsh_onall(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_all_not_harsh'] = res_all_nh
    
    i2f_res_all = i2f_topk_not_harsh(root=root, model_to_evaluate_path=model_to_evaluate_path, test_data_path=test_data_path,target_batch_path=target_data_path, save_CV_dir=save_CV_dir, k=k)
    results['i2f_all']= i2f_res_all

 
    #!rm -r /users/visics/fbabilon/Sherlock/export_Sherlock/CV_feat/cvae_from_B1_elastic/reg05/*
    print(' %s B1'%str(name))
    target_data_path = root+ '/data_splits/B1_test.cvs'
    
    res_b1_onall=eval_map_not_harsh_onall(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_b1_onall'] = res_b1_onall
    
    res_b1_oneach=eval_map_not_harsh_oneach(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_b1_oneach'] = res_b1_oneach

    
    i2f_res_b1 = i2f_topk_not_harsh(root=root, model_to_evaluate_path=model_to_evaluate_path, test_data_path=test_data_path,target_batch_path=target_data_path, save_CV_dir=save_CV_dir, k=k)
    results['i2f_res_b1'] = i2f_res_b1

    print(' %s B2'%str(name))
    target_data_path=root+ '/data_splits/B2_test.cvs'
    
    res_b2_onall=eval_map_not_harsh_onall(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_b2_onall'] = res_b2_onall   
    res_b2_oneach=eval_map_not_harsh_oneach(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_b2_oneach'] = res_b2_oneach
    
     
    i2f_res_b2 = i2f_topk_not_harsh(root=root, model_to_evaluate_path=model_to_evaluate_path, test_data_path=test_data_path,target_batch_path=target_data_path, save_CV_dir=save_CV_dir, k=k)
    results['i2f_res_b2'] = i2f_res_b2


    print('Saved results.')
    torch.save(results, save_res)

####################################################################################################
####################################################################################################
def eval_multiple_Tasks_not_duplicate_i2f_and_f2i(exp_res, model_to_evaluate_path, root, test_data_path, save_CV_dir, k, list_target_path): #print(model_to_evaluate_path, name)
    r"""
    Computes the remaining evaluation metric for multiple tasks jobs
    """
    results={}
    
    if os.path.exists(save_CV_dir):
        shutil.rmtree(save_CV_dir)
    
    #if  not os.path.exists(exp_res):
    #    os.makedirs(exp_res)
    
    
    print(' %s ALL'%str(exp_res))
    target_data_path = root+ 'data_info/test_SPO_df.cvs'    
    res_all=eval_map(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_all_harsh'] = res_all
    res_all_nh=eval_map_not_harsh_onall(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_all_not_harsh'] = res_all_nh
    
    i2f_res_all = i2f_topk_not_harsh(root=root, model_to_evaluate_path=model_to_evaluate_path, test_data_path=test_data_path,target_batch_path=target_data_path, save_CV_dir=save_CV_dir, k=k)
    results['i2f_all']= i2f_res_all
    

    c=1
    for target_data_path in list_target_path:

        target_name = 'res_t%s_'%str(c) 
        c +=1
        print('%s'%str(target_name))
        res_onall=eval_map_not_harsh_onall(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
        results[ target_name + 'onall'] = res_onall

        res_oneach=eval_map_not_harsh_oneach(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
        results[target_name + 'oneach'] = res_oneach

        
        i2f_res = i2f_topk_not_harsh(root=root, model_to_evaluate_path=model_to_evaluate_path, test_data_path=test_data_path,target_batch_path=target_data_path, save_CV_dir=save_CV_dir, k=k)
        results[target_name +'i2f'] = i2f_res
        
        


    try:
        torch.save(results, exp_res)
        print('Saved results.')
    except:
        pdb.set_trace()

####################################################################################################
def eval_four_Tasks_not_duplicate_i2f_and_f2i(name, model_to_evaluate_path, root, test_data_path, save_CV_dir, k, list_target_path): #print(model_to_evaluate_path, name)
    r"""
    Computes the remaining evaluation metric for 2tasks jobs
    """
    
    if os.path.exists(save_CV_dir):
        shutil.rmtree(save_CV_dir)
    
    save_res= root + '/code/results/%s'%str(name)

    results={}
    
    
    print(' %s ALL'%str(name))
    target_data_path = root+ 'data_info/test_SPO_df.cvs'    
    res_all=eval_map(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_all_harsh'] = res_all
    res_all_nh=eval_map_not_harsh_onall(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
    results['res_all_not_harsh'] = res_all_nh
    
    i2f_res_all = i2f_topk_not_harsh(root=root, model_to_evaluate_path=model_to_evaluate_path, test_data_path=test_data_path,target_batch_path=target_data_path, save_CV_dir=save_CV_dir, k=k)
    results['i2f_all']= i2f_res_all
    

    c=1
    for target_data_path in list_target_path:

        target_name = 'res_b%s_'%str(c) 
        c +=1
        print(name,'%s'%str(target_name))
        res_onall=eval_map_not_harsh_onall(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
        results[ target_name + 'onall'] = res_onall

        res_oneach=eval_map_not_harsh_oneach(target_batch_path=target_data_path,test_data_path=test_data_path,model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
        results[target_name + 'oneach'] = res_oneach


        i2f_res = i2f_topk_not_harsh(root=root, model_to_evaluate_path=model_to_evaluate_path, test_data_path=test_data_path,target_batch_path=target_data_path, save_CV_dir=save_CV_dir, k=k)
        results[target_name +'i2f'] = i2f_res


    print('Saved results.')
    torch.save(results, save_res)
################################################################################################################################
#ECCV
def eval_map_with_results(root=None, model_to_evaluate_path=None, test_data_path=None,target_batch_path=None, save_CV_dir=None, NLP_feat_path=None, unique_facts_id_path=None, batch=None, score_type=None, ascending_flag=None, k=None, dict_pairs_path=None, use_gpu=None,use_multiple_gpu=None):
    r"""
    assumes target_batch_path=test_data_path
    """
    
    'DEINE YOUR DIRECTORIES'
    root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path = fill_default_vals(root, model_to_evaluate_path, test_data_path,target_batch_path, save_CV_dir, NLP_feat_path, unique_facts_id_path, batch, score_type, ascending_flag, k, dict_pairs_path)
    df_test= pd.read_csv(test_data_path)
    
    #make your save directory
    if not os.path.exists(os.path.join(save_CV_dir, 'feats.npy')):
        print('extracting the features')
        # os.makedirs(save_CV_dir)
        resume_path=model_to_evaluate_path
        Sherlock_Net, checkpoint = train_eval_utils.load_model('',model_to_evaluate_path, use_gpu,use_multiple_gpu)
        dset_loaders=dataset_utils.load_eval_dataset(root, test_data_path) 
        print('Dataset Loaded. your batch size is %s'%batch)
        CV_feat=train_eval_utils.eval_fact_model_npy(Sherlock_Net, dset_loaders, root, save_CV_dir) 
        np.save(os.path.join(save_CV_dir,'feats'), CV_feat)
    else:
        CV_feat=np.load(os.path.join(save_CV_dir, 'feats.npy'))
        print('Feats Loaded.')
    feat_df=df_test
    #load the Unique facts dataframe
    facts_df = pd.read_csv(unique_facts_id_path) 
    
    #select from the list of unique ids present in your test images  
    Batch_facts=pd.read_csv(target_batch_path)
    facts_to_evaluate=(sorted(list(set(Batch_facts['id']))))
    
    #for each fact, get a mAP score over k images
    py_mAp_k=[] 
    for fact_id in facts_to_evaluate:
        #pdb.set_trace()
        mAp_k =performance_utils.ap_k(fact_id, k, score_type, CV_feat, feat_df, facts_df, root, ascending_flag )
        # print (fact_id, 'mAP100:%s'%str(mAp_k))
        py_mAp_k.append(mAp_k)

    #then print as evaluation metric the average mAP100    
    result=np.mean(np.asarray(py_mAp_k))
    print('Averaged mAP%s:'%str(k), result)
    #save mAP for each id
    d={'id':facts_to_evaluate, 'mAP%s'%str(k):py_mAp_k}
    df_mAp_k=pd.DataFrame(data=d)
    
    return result, df_mAp_k
