# Generated with SMOP  0.41
# from libsmop import *
from GetResults_exact_metric1 import *
import os
import scipy.io
import json
from pprint import pprint

root='/home/abdelksa/c2044/lifelong_fact_learning/'
model_names = os.listdir(root + '/outputs/CV_feat')
model_names = sorted(model_names)
CV_dir = root + '/outputs/CV_feat/'
tembedding_path= root + 'eval/eval_files/TEmbeddings/large_scale/semantic/'

for model_name in model_names:
    results = []
    if '8tasks_semantic' in model_name:
        if not all([os.path.exists(CV_dir + model_name + '/T8' + '/B' + str(b) + 'XEmbeddings.mat') for b in range(1, 9)]):
            continue
        print(model_name)
        for b in range(1, 9):
            X_embedding=scipy.io.loadmat(root + 'outputs/CV_feat/' + model_name + '/T8/B' + str(b) + 'XEmbeddings.mat')
            X_embedding=X_embedding['XE']
            T_embedding=scipy.io.loadmat(tembedding_path + 'TEmbedding_rnd_B' + str(b) + '_test.mat')
            T_embedding=T_embedding['B_T']
            TestData=scipy.io.loadmat(tembedding_path +'TestData_rnd_B' + str(b) + '_test.mat')
            TestData=TestData['BTestData']
            res_name= root + 'results/lrs_results/semantic/' + model_name + '_batch' + str(b) + '_m1_nodup.mat'
            Result_exact=GetResults_exact_metric1(X_embedding,T_embedding,TestData,T_embedding,flag_nodup=True)
            # pprint(Result_exact['mAP'])
            results.append('{:.2f}%'.format(Result_exact['KnowledgeMeanDetRatio_K5'] * 100.))
            # print('T{}:'.format(b), '{:.2f}%'.format(Result_exact['KnowledgeMeanDetRatio_K5'] * 100.))
            # exit()
            # save(res_name,'Result_exact') # Todo: save as json
        print('T1, T2, T3, T4, T5, T6, T7, T8, mean:')
        print(*results, np.mean(results), sep='\t')
