# Generated with SMOP  0.41
# from libsmop import *
from GetResults_exact_metric1 import *
import os
import scipy.io
import json
from pprint import pprint

# addpath(genpath('./APcode'))
# addpath(genpath('./AUCcode'))
root='/Users/sherifabdelkarim/projects/lifelong_learning_raw_files/eval'
model_names = os.listdir(root + '/CV_feat')
model_names = sorted(model_names)
# model_names = ['finetune_4tasks_random_reg0_lr3e-05']
# tembedding_path='/Users/sherifabdelkarim/projects/lifelong_learning_raw_files/eval/MDS/MD_random_eval/Temb_MD_random/'

for model_name in model_names:
    if '8tasks_semantic' in model_name:
        print(model_name)
        for b in range(1, 9):
            X_embedding=scipy.io.loadmat(root + '/CV_feat/' + model_name + '/B' + str(b) + 'XEmbeddings.mat')
            X_embedding=X_embedding['XE']
            T_embedding=scipy.io.loadmat(root + '/semantic_eval/TEmbedding_rnd_B' + str(b) + '_test.mat')
            T_embedding=T_embedding['B_T']
            TestData=scipy.io.loadmat(tembedding_path +'/semantic_eval/TestData_rnd_B' + str(b) + '_test.mat')
            TestData=TestData['BTestData']
            res_name= root + '/results/lrs_results/semantic/' + model_name + '_batch' + str(b) + '_m1_nodup.mat'
            Result_exact=GetResults_exact_metric1(X_embedding,T_embedding,TestData,T_embedding,flag_nodup=True)
            # pprint(Result_exact['mAP'])
            print('T{}:'.format(b), '{:.2f}%'.format(Result_exact['KnowledgeMeanDetRatio_K5'] * 100.))
            # exit()
            # save(res_name,'Result_exact') # Todo: save as json
            # json.
