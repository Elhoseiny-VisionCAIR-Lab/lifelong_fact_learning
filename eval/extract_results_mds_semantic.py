# Generated with SMOP  0.41
# from libsmop import *
from GetResults_exact_metric1 import *
import os
import scipy.io
import json
from pprint import pprint

# addpath(genpath('./APcode'))
# addpath(genpath('./AUCcode'))
root='/home/abdelksa/c2044/lifelong_fact_learning/'
model_names = os.listdir(root + '/outputs/CV_feat')
model_names = sorted(model_names)
# model_names = ['finetune_4tasks_random_reg0_lr3e-05']
tembedding_path= root + 'eval/TEmbeddings/mid_scale/semantic/Temb_MD_semantic/'

for model_name in model_names:
    if '4tasks_semantic' in model_name:
        print(model_name)
        for b in range(1, 5):
            X_embedding=scipy.io.loadmat(root + '/outputs/CV_feat/' + model_name + '/B' + str(b) + 'XEmbeddings.mat')
            X_embedding=X_embedding['XE']
            T_embedding=scipy.io.loadmat(tembedding_path + '/TEmbedding_seen_B' + str(b) + '_test.mat')
            T_embedding=T_embedding['B_T']
            TestData=scipy.io.loadmat(tembedding_path +'/TestData_seen_B' + str(b) + '_test.mat')
            TestData=TestData['BTestData']
            res_name= root + '/eval/results/mds_results/semantic/' + model_name + '_batch' + str(b) + '_m1_nodup.mat'
            Result_exact=GetResults_exact_metric1(X_embedding,T_embedding,TestData,T_embedding,flag_nodup=True)
            # pprint(Result_exact['mAP'])
            print('T{}:'.format(b), '{:.2f}%'.format(Result_exact['KnowledgeMeanDetRatio_K5'] * 100.))
            # exit()
            # save(res_name,'Result_exact') # Todo: save as json
