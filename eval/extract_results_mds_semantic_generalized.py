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
tembedding_path= root + 'eval/eval_files/TEmbeddings/mid_scale/semantic/'

for model_name in model_names:
    if '4tasks_semantic' in model_name:
        # if '4tasks_v' in model_name:
        lens = []
        if not all([os.path.exists(CV_dir + model_name + '/T4' + '/B' + str(b) + 'XEmbeddings.mat') for b in range(1, 5)]):
        # if not all([os.path.exists(CV_dir + model_name + '/B' + str(b) + 'XEmbeddings.mat') for b in range(1, 5)]):
            continue
        print('\n' + model_name)

        for b in range(1, 5):
            mat_path = CV_dir + model_name + '/T4' + '/B' + str(b) + 'XEmbeddings.mat'
            # mat_path = CV_dir + model_name + '/B' + str(b) + 'XEmbeddings.mat'
            X_embedding = scipy.io.loadmat(mat_path)
            X_embedding = X_embedding['XE']
            lens.append(X_embedding.shape[0])
            T_embedding = scipy.io.loadmat(tembedding_path + 'TEmbedding_seen_B' + str(b) + '_test.mat')
            T_embedding = T_embedding['B_T']
            TestData = scipy.io.loadmat(tembedding_path + 'TestData_seen_B' + str(b) + '_test.mat')
            TestData = TestData['BTestData']
            res_name = root + 'results/mds_results/semantic/' + model_name + '_batch' + str(b) + '_generalized_nodup.mat'
            if b == 1:
                X_embedding_all = X_embedding
                T_embedding_all = T_embedding
                TestData_all = TestData
            else:
                X_embedding_all = np.concatenate([X_embedding_all, X_embedding], axis=0)
                T_embedding_all = np.concatenate([T_embedding_all, T_embedding], axis=0)
                # for key in ['im_ids', 'im_names', 'tuple_ids', 'labels', 'unique_tuple_ids', 'unique_tuple_features', '', '']
                TestData_all['im_ids'][0][0] = np.concatenate(
                    [TestData_all['im_ids'][0][0], TestData['im_ids'][0][0]], axis=1)
                TestData_all['im_names'][0][0] = np.concatenate(
                    [TestData_all['im_names'][0][0], TestData['im_names'][0][0]], axis=1)
                TestData_all['tuple_ids'][0][0] = np.concatenate(
                    [TestData_all['tuple_ids'][0][0], TestData['tuple_ids'][0][0]], axis=1)
                TestData_all['labels'][0][0] = np.concatenate(
                    [TestData_all['labels'][0][0], TestData['labels'][0][0]], axis=1)
                TestData_all['unique_tuple_ids'][0][0] = np.concatenate(
                    [TestData_all['unique_tuple_ids'][0][0], TestData['unique_tuple_ids'][0][0]], axis=1)
                TestData_all['unique_tuple_features'][0][0] = np.concatenate(
                    [TestData_all['unique_tuple_features'][0][0], TestData['unique_tuple_features'][0][0]],
                    axis=0)
                TestData_all['unique_tuple_ids'][0][0], ia, ic = np.unique(
                    TestData_all['unique_tuple_ids'][0][0], return_inverse=True, return_index=True)
                TestData_all['unique_tuple_ids'][0][0] = np.expand_dims(TestData_all['unique_tuple_ids'][0][0], axis=0)
                TestData_all['unique_tuple_features'][0][0] = TestData_all['unique_tuple_features'][0][0][ia]

        Result_exact = GetResults_exact_metric1(X_embedding_all,
                                                T_embedding_all,
                                                TestData_all,
                                                T_embedding_all,
                                                flag_nodup=True)
        print('T1, T2, T3, T4, mean:')
        print('{:.2f}%'.format(np.mean(Result_exact['KnowledgeDetRatios_K5'][:lens[0]]) * 100) + '\t' +
              '{:.2f}%'.format(np.mean(Result_exact['KnowledgeDetRatios_K5'][lens[0]:sum(lens[:2])]) * 100) + '\t' +
              '{:.2f}%'.format(np.mean(Result_exact['KnowledgeDetRatios_K5'][sum(lens[:2]):sum(lens[:3])]) * 100) + '\t' +
              '{:.2f}%'.format(np.mean(Result_exact['KnowledgeDetRatios_K5'][sum(lens[:3]):]) * 100) + '\t' +
              '{:.2f}%'.format(Result_exact['KnowledgeMeanDetRatio_K5'] * 100.))
