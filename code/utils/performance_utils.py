#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 19:49:54 2017

@author: fra
"""
import pandas as pd
import numpy as np
import pdb
import scipy.spatial.distance as spd
from sklearn.metrics import roc_auc_score as ROC
import numbers
from PIL import Image
from torchvision import transforms
from scipy.spatial import distance
import os

######################################################################################################################
#functions
def feat_loader(path):
    try:
        feat=np.loadtxt(path)
    except ValueError:
        pdb.set_trace()
    return feat

def get_NLP_feat(facts_df, fact_id, root):
    #collect your NLP_fact feat (1, 900)    
    fact_path=facts_df[facts_df['id']==fact_id]['NLP_links']        
    #pdb.set_trace()
    fact_NLP_feat=feat_loader(root + 'NLP_feat/unique_facts/' + str(fact_path).split()[1])  
    
    return fact_NLP_feat

def Scorer(fact_id, score_type, images_CV_feat, facts_df, root):
    '''
    input : Fact_id = fact that you are considering (each SPO has its own fact_id)
            Score_type: the distance metric that you want to compute
            
    output: score for each image in the test set:
            distance between CV_image_feat and NLP_fact_feat according to the selected metric
    
    '''
    
    fact_NLP_feat= get_NLP_feat(facts_df, fact_id,root)

    
    #Compute 1 score per image
    if score_type == 'dot':
        print('Computing %s Score '%score_type)
        dot_score = np.dot(images_CV_feat, fact_NLP_feat)
        
        '''
        #test against a random NLP feat
        rand_NLP=np.random.rand(900)
        dot_score_rnd=np.dot(images_CV_feat, fact_NLP_feat)
        
        #give back results
        data={'score': dot_score,
             'rnd_test_score': dot_score_rnd}
        return pd.DataFrame(data, columns=['%s_score'%score_type, '%s_score_random'%score_type])
        '''
        return pd.DataFrame(dot_score, columns=['%s_score'%score_type])
        
    if score_type == 'cos':
        print('Computing %s Score '%score_type)
        fact_NLP_feat= fact_NLP_feat.reshape(1, len(fact_NLP_feat)) #without reshape gives (900,) as dim.
        
        cos_score = 1 - spd.cdist(images_CV_feat, fact_NLP_feat, 'cosine')
        return pd.DataFrame(cos_score, columns=['%s_score'%score_type])
    
    if score_type == 'euc':
        #print('Computing fancy version of the %s Score '%score_type)
        fact_NLP_feat= fact_NLP_feat.reshape(1, len(fact_NLP_feat)) #without reshape gives (900,) as dim.
        
        naive_euc_score = spd.cdist(images_CV_feat, fact_NLP_feat, 'euclidean')
        sigma=  np.median(naive_euc_score)
        euc_score = np.exp(-1*(np.power(naive_euc_score,2)/ float(2* np.power(sigma,2)))) 
        return pd.DataFrame(euc_score, columns=['%s_score'%score_type])
     
    if score_type == 'plain_euc':
        print('Computing standard version of the %s Score '%score_type)
        fact_NLP_feat= fact_NLP_feat.reshape(1, len(fact_NLP_feat)) #without reshape gives (900,) as dim.
        
        euc_plain_score = spd.cdist(images_CV_feat, fact_NLP_feat, 'euclidean')
        
        
        #return results as pandas df
        return pd.DataFrame(euc_plain_score, columns=['%s_score'%score_type])


#get score and gts
def get_score_and_gts(fact_id, score_type, images_CV_feat, feat_df, facts_df, root):
    '''
    input: fact_id :   the fact id  
           score_type: the type of distance that you want to use to compute how far are NLP_fact & CV_feat 
           
    output: df  : 'score' according to your score_type
                : 'gts' according to your selected fact
    '''        
    scores = Scorer(fact_id, score_type, images_CV_feat, facts_df, root)
    # gts    = (feat_df['id']==fact_id).as_matrix().astype(int)  # len(gt)=num_images, 0= image DO NOT belong to fact_id, 1=image DO belong to fact_id
    gts    = (feat_df['id']==fact_id).to_numpy().astype(int)  # len(gt)=num_images, 0= image DO NOT belong to fact_id, 1=image DO belong to fact_id
    gts=pd.DataFrame(gts, columns=['gts_for_fact:%s'%fact_id])
    
    # print('Getting results for fact_id: %s'%fact_id)
    results=pd.concat([scores,gts], axis=1)
    
    return results

#get score and gts
def get_score_and_gts_and_images(fact_id, score_type, images_CV_feat, feat_df, facts_df, root):
    '''
    input: fact_id :   the fact id  
           score_type: the type of distance that you want to use to compute how far are NLP_fact & CV_feat 
           
    output: df  : 'score' according to your score_type
                : 'gts' according to your selected fact
    '''        
    scores = Scorer(fact_id, score_type, images_CV_feat, facts_df, root)
    gts    = (feat_df['id']==fact_id).as_matrix().astype(int)  # len(gt)=num_images, 0= image DO NOT belong to fact_id, 1=image DO belong to fact_id
    gts=pd.DataFrame(gts, columns=['gts_for_fact:%s'%fact_id])
    
    # print('Getting results for fact_id: %s'%fact_id)
    results=pd.concat([scores,gts], axis=1)
    
    feat_df['scores'] = scores
    feat_df['gts'] = gts
    #pdb.set_trace()
    return feat_df

#sanity check
#df=get_score_and_gts(0,'cos')
    
def precision(fact_id, sort_df,k):   
    #get the number of true positive in the firt k ranked images
    ntp = sort_df[0:k]['gts_for_fact:%s'%fact_id].sum() 
    
    #sanity check
    #if k % 100:
    #   print ('TRue Positive for first %d images:'%k, ntp)
    
    #compute precision
    if (ntp==0):
        precision=ntp
    else:
        precision= float(ntp / float(k)) 
        
    return ntp, precision


def copied_precision(fact_id, sort_df, k):   
    #get the number of true positive in the firt k ranked images
    pool = sort_df[0:k]['gts_for_fact:%s'%fact_id] 
    precisions = []
    nTP = 0
    for i in range(k):
        gt = pool.iloc[i]
        if gt==1:
            nTP = nTP + 1
            precisions.append(float(nTP/float(i+1)))
           
    if nTP == 0:
         ap=0
         pn=0
    else:
         ap = float(sum(precisions)/ float(nTP))
         pn = precisions[-1]
    
    return ap, pn

#get ranking metrics
def ap_k(fact_id, k, score_type, images_CV_feat, feat_df, facts_df, root, ascending_flag=0):
    '''
    input:  fact_id, score_type
            k = num ranking images that you want to consider
            ascending_flag=0: the higher the better (this is the right flag for euc score only, plain_euC, dot and cos should have the flag set to 1) 
    
    
    output: ntp  = number of true positive in your K highest renked images
            AP_k = 
    '''
    
    
    #get scores, gts
    df=get_score_and_gts(fact_id, score_type, images_CV_feat, feat_df, facts_df, root) 
    
    #sort df according to scores 
    sort_df=df.sort_values(by='%s_score'%score_type, ascending=ascending_flag)   
    
    #get average precision over the frist k images
    # AP_K=[]                         # TODO: use sklearn function for average precision
    # for i in range(k):
    #     ap = copied_precision(fact_id, sort_df, i)
    #     #sanity check
    #     #if i % 100:
    #     #    print('Precision for first %d images:'%i, precision_i)
    #     AP_K.append(ap)
        
    # mAP_K = np.mean(np.asarray(AP_K))
    AP_K = copied_precision(fact_id, sort_df, k)
    return AP_K


###############################################################################################
#NOT HARSH METRIC 
    
#Find facts ids that have the same NLP feat. 
def find_dataset_pairs(facts, root):
    r"""
    find pairs in your w2v representation,saving a dictionary with all the repeted facts
    input: facts df
    output: pair dictionary
            - keys is new label proposal - is the id of the first item in "facts_df" with that specific nlp feat 
            - arg are list of facts that shared the w2v representation with the key.
            
            for example:
            pairs[62] = [62, 104, 162]

    """
    
    #build w2v unique_facts_matrix
    flag=False
    for index, row in facts.iterrows():
        feat=np.loadtxt(root + 'NLP_feat/unique_facts/'+ row['NLP_links'])
        if flag :
            matrix= np.vstack((matrix,feat))

        else:
            matrix=feat
            flag=True
            
    print('NLP matrix built')
    
    #compute the distance between all unique facts w2v.
    #pdist will give you a condensed triangular matrix with the distance for each pair f1,f2
    x=distance.pdist(matrix)

    #You iterate over each fact pair f1=row & f2=col , filling a pair dictionary 
    i=0
    pairs={}

    #for each f1
    for row in range(0, (len(facts)-1)):

        #check if f1 has some pairs with every other fact id (pair means same w2v representation)
        for col in range(row+1, len(facts)): 
            if x[i]==0:
                #if you already have a match don't do anything
                if col in [x for v in pairs.values() for x in v]:
                    print('you should have this pair', col, row)
                    
                #if you have an addiction to a prior match, append the new id there
                else:
                    if row in pairs:
                        pairs[row].append(col) 
                        #print('addiction to key!', row)
                        #print(pairs)
                    
                    #otherwise extend the dict
                    else:
                        pairs[row]=[row, col]
                        #print('new key!', row)
                        #print(pairs)
    
            i +=1
        
        #as sanity check print some results
        #if (len(f1_pairs) > 1) and (row<100):
           # print('Repetition of Fact:%s'%str(row))
           # print(facts[facts['id'].isin(f1_pairs)])

    return pairs


def merge_duplicate(new_facts, new_test, pairs):
    r"""
    merge duplicate ids from facts, test_df 
    merging every fact that share a w2v representation under the same id
    """
    
    #Identify facts pair
    print('Duplicate dict:', pairs)
    
    #Change facts ids
    facts_ids=new_facts['id']
    for k in pairs.keys(): # 1: 2, 5, 6, 7
        #w.loc[w.female != 'female', 'female'] = 0
        facts_ids = facts_ids.replace(pairs[k], k)
        #print('facts',pairs[k], 'will be merged with fact:%s'%str(k))

    new_facts['id']=facts_ids
    

    #Change test ids
    test_ids=new_test['id']
    for k in pairs.keys():
        #w.loc[w.female != 'female', 'female'] = 0
        test_ids = test_ids.replace(pairs[k], k)

    new_test['id']=test_ids
    
##################NOT  HARSH METRIC PT 2#############################################################################
#get score and gts
def get_score_and_not_harsh_gts(fact_id, score_type, images_CV_feat, feat_df, facts_df, root):
    '''
    input: fact_id :   the fact id  
           score_type: the type of distance that you want to use to compute how far are NLP_fact & CV_feat 
           
    output: df  : 'score' according to your score_type
                : 'gts' according to your selected fact
    '''        
    scores = Scorer(fact_id, score_type, images_CV_feat, facts_df, root)
    #Check if the item is an S, SP or SPO. Assign as gt=1 all the facts that match that representation
    #if fact_type='S' every fact that has that specific 'S' is gt=1
    #if fact_type='SP' every fact that has that specific 'SP' is gt=1
    #if fact_type='SPO' every fact that has that specific 'SPO' is gt=1
    pdb.set_trace()
    gts    = (feat_df['id']==fact_id).as_matrix().astype(int)  
    gts=pd.DataFrame(gts, columns=['gts_for_fact:%s'%fact_id])
    
    # print('Getting results for fact_id: %s'%fact_id)
    results=pd.concat([scores,gts], axis=1)
    
    return results


#get ranking metrics
def ap_k_not_harsh(fact_id, k, score_type, images_CV_feat, feat_df, facts_df, root, ascending_flag=0):
    '''
    input:  fact_id, score_type
            k = num ranking images that you want to consider
            ascending_flag=0: the higher the better (this is the right flag for euc score only, plain_euC, dot and cos should have the flag set to 1) 
    
    
    output: ntp  = number of true positive in your K highest renked images
            AP_k = 
    '''
    
    
    #get scores, gts NOT HARSH
    df=get_score_and_not_harsh_gts(fact_id, score_type, images_CV_feat, feat_df, facts_df, root) 
    
    #sort df according to scores 
    sort_df=df.sort_values(by='%s_score'%score_type, ascending=ascending_flag)   
    
    #get average precision over the frist k images
    # AP_K=[]
    # for i in range(k):
    #     ap = copied_precision(fact_id, sort_df, i)
    #     #sanity check
    #     #if i % 100:
    #     #    print('Precision for first %d images:'%i, precision_i)
    #     AP_K.append(ap)
        
    # mAP_K = np.mean(np.asarray(AP_K))
    AP_K = copied_precision(fact_id, sort_df, k)
    return AP_K


##################FROM IMAGE TO FACTS#################################################################################

def i2f_scorer(image_CV_path, score_type, NLP_feats, facts_df):
    '''
    input : Fact_id = fact that you are considering (each SPO has its own fact_id)
            Score_type: the distance metric that you want to compute
            
    output: score for each image in the test set:
            distance between CV_image_feat and NLP_fact_feat according to the selected metric
    
    '''
    
    image_CV_feat = np.loadtxt( image_CV_path)

    
    #Compute 1 score per image
    if score_type == 'dot':
        print('Computing %s Score '%score_type)
        dot_score = np.dot(NLP_feats, image_CV_feat)
        
        '''
        #test against a random NLP feat
        rand_NLP=np.random.rand(900)
        dot_score_rnd=np.dot(images_CV_feat, fact_NLP_feat)
        
        #give back results
        data={'score': dot_score,
             'rnd_test_score': dot_score_rnd}
        return pd.DataFrame(data, columns=['%s_score'%score_type, '%s_score_random'%score_type])
        '''
        return pd.DataFrame(dot_score, columns=['%s_score'%score_type])
        
    if score_type == 'cos':
        print('Computing %s Score '%score_type)
        image_CV_feat= image_CV_feat.reshape(1, len(image_CV_feat)) #without reshape gives (900,) as dim.
        
        cos_score = 1 - spd.cdist(NLP_feats, image_CV_feat, 'cosine')
        return pd.DataFrame(cos_score, columns=['%s_score'%score_type])
    
    if score_type == 'euc':
        #print('Computing fancy version of the %s Score '%score_type)
        image_CV_feat= image_CV_feat.reshape(1, len(image_CV_feat)) #without reshape gives (900,) as dim.
        
        naive_euc_score = spd.cdist(NLP_feats, image_CV_feat, 'euclidean')
        sigma=  np.median(naive_euc_score)
        euc_score = np.exp(-1*(np.power(naive_euc_score,2)/ float(2* np.power(sigma,2)))) 
        #return euc_score
        return pd.DataFrame(euc_score, columns=['%s_score'%score_type])
     
    if score_type == 'plain_euc':
        print('Computing standard version of the %s Score '%score_type)
        image_CV_feat= image_CV_feat.reshape(1, len(image_CV_feat)) #without reshape gives (900,) as dim.
        
        euc_plain_score = spd.cdist(NLP_feats, image_CV_feat, 'euclidean')
        
        
        #return euc_plain_score
        return pd.DataFrame(euc_plain_score, columns=['%s_score'%score_type])


    
