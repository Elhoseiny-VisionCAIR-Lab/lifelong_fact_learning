# Generated with SMOP  0.41
# from libsmop import *
from get_exact_cos import get_exact_cos
import numpy as np
import os
from scipy.spatial import distance
from APcode.calcAP import calcAP

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return np.array([1 if itm in b else 0 for itm in a]), np.array([bind.get(itm, None) for itm in a])


# from smop.libsmop import *
# GetResults_exact_metric1_python.m

# X_embedding: embedding of testing images N x D
# T_embedding: embedding of tesing facts  N x D
# X_embedding,  T_embedding is the embedding in the common space for each test pair. This is different based on each method that perform common embedding
# TestData specs (includes data like file names, ground truth data, etc )
# We use FLANN librarypip
# T_embedding_900: N x D2 . It is used mainly for the purpose to get ground truth annotation together with TestData.  This array is fixed for all methods. This is just to check facts of same subject, same subject and predicate, and same subject predicate and objects.

def GetResults_exact_metric1(X_embedding=None, T_embedding=None, TestData=None, T_embedding_900=None, flag_nodup=False):
    # varargin = GetResults_exact_metric1.varargin
    # nargin = GetResults_exact_metric1.nargin

    tmp_prefix = 'metric1_'
    # GetResults_exact_metric1_python.m:12
    #     if (~exist('flag_always_compute','var')):
    flag_always_compute = 1
    # GetResults_exact_metric1_python.m:15

    # GetResults_exact_metric1_python.m:19
    #     X_embedding=double(X_embedding)
    # GetResults_exact_metric1_python.m:21
    #     tupleId2IndMap=containers.Map(TestData.unique_tuple_ids,concat([arange(1,len(TestData.unique_tuple_ids))]))
    # GetResults_exact_metric1_python.m:22
    #     im_names_unique,ia_imnames,ic_imnames=unique(TestData.im_names,nargout=3)
    #     print(TestData['im_names'][0][0].shape)
    im_names_unique, ia_imnames, ic_imnames = np.unique(TestData['im_names'][0][0], return_index=True,
                                                        return_inverse=True)
    # print(im_names_unique.shape, ia_imnames.shape, ic_imnames.shape)

    # GetResults_exact_metric1_python.m:24

    # computed T_embedding mean has the mean subtracted twice
    # T_embedding_900 = bsxfun(@plus,T_embedding_900,TestData.unique_tuple_features_mean);
    # T_unique_embeddings_900,ia,ic=unique(T_embedding_900,'rows',nargout=3)
    T_unique_embeddings_900, ia, ic = np.unique(T_embedding_900, return_index=True, return_inverse=True, axis=0)
    # print(T_unique_embeddings_900.shape, ia.shape, ic.shape)
    # GetResults_exact_metric1_python.m:28
    #     T_unique_embeddings=T_embedding(ia,arange())
    T_unique_embeddings = T_embedding[ia, :]
    # GetResults_exact_metric1_python.m:29
    #     T_unique_embeddings_plus_mean=bsxfun(plus,T_unique_embeddings_900,TestData.unique_tuple_features_mean)
    T_unique_embeddings_plus_mean = T_unique_embeddings_900 + TestData['unique_tuple_features_mean'][0][0]

    # GetResults_exact_metric1_python.m:31
    T_unique_embeddings_plus_mean_Sq = T_unique_embeddings_plus_mean ** 2
    # GetResults_exact_metric1_python.m:33
    T_unique_embeddings_plus_mean_S_equal0 = (
            np.sum(T_unique_embeddings_plus_mean_Sq[:, :300], axis=1) == 0)  # .astype(int)
    # GetResults_exact_metric1_python.m:34
    T_unique_embeddings_plus_mean_P_equal0 = (
            np.sum(T_unique_embeddings_plus_mean_Sq[:, 300:600], axis=1) == 0)  # .astype(int)
    # GetResults_exact_metric1_python.m:35
    T_unique_embeddings_plus_mean_O_equal0 = (
            np.sum(T_unique_embeddings_plus_mean_Sq[:, 600:], axis=1) == 0)  # .astype(int)
    # GetResults_exact_metric1_python.m:36
    num_S_0 = sum(T_unique_embeddings_plus_mean_S_equal0)
    # GetResults_exact_metric1_python.m:38
    num_P_0 = sum(T_unique_embeddings_plus_mean_P_equal0)
    # GetResults_exact_metric1_python.m:39
    num_O_0 = sum(T_unique_embeddings_plus_mean_O_equal0)
    # GetResults_exact_metric1_python.m:40
    K = 100
    # print((T_unique_embeddings_plus_mean_S_equal0 == np.logical_and(0,T_unique_embeddings_plus_mean_P_equal0)) == np.logical_and(1,T_unique_embeddings_plus_mean_O_equal0))

    # GetResults_exact_metric1_python.m:42
    # Is_FirstOrder = ((T_unique_embeddings_plus_mean_S_equal0 == np.logical_and(0, T_unique_embeddings_plus_mean_P_equal0))
    #                  == np.logical_and(1, T_unique_embeddings_plus_mean_O_equal0)) == 1

    # Is_FirstOrder = T_unique_embeddings_plus_mean_S_equal0 == 0 & T_unique_embeddings_plus_mean_P_equal0 == 1 & T_unique_embeddings_plus_mean_O_equal0 == 1;
    Is_FirstOrder = np.logical_and(np.logical_and(
        np.equal(T_unique_embeddings_plus_mean_S_equal0, 0),
        np.equal(T_unique_embeddings_plus_mean_P_equal0, 1)),
        np.equal(T_unique_embeddings_plus_mean_O_equal0, 1))
    # GetResults_exact_metric1_python.m:45
    # Is_SecondOrder = ((T_unique_embeddings_plus_mean_S_equal0 == np.logical_and(0, T_unique_embeddings_plus_mean_P_equal0))
    #                   == np.logical_and(0, T_unique_embeddings_plus_mean_O_equal0)) == 1
    Is_SecondOrder = np.logical_and(np.logical_and(
        np.equal(T_unique_embeddings_plus_mean_S_equal0, 0),
        np.equal(T_unique_embeddings_plus_mean_P_equal0, 0)),
        np.equal(T_unique_embeddings_plus_mean_O_equal0, 1))

    # GetResults_exact_metric1_python.m:47
    # Is_ThirdOrder = ((T_unique_embeddings_plus_mean_S_equal0 == np.logical_and(0, T_unique_embeddings_plus_mean_P_equal0))
    #                  == np.logical_and(0, T_unique_embeddings_plus_mean_O_equal0)) == 0

    Is_ThirdOrder = np.logical_and(np.logical_and(
        np.equal(T_unique_embeddings_plus_mean_S_equal0, 0),
        np.equal(T_unique_embeddings_plus_mean_P_equal0, 0)),
        np.equal(T_unique_embeddings_plus_mean_O_equal0, 0))


    Result = dict()
    # GetResults_exact_metric1_python.m:50
    Result['Is_FirstOrder'] = Is_FirstOrder
    # GetResults_exact_metric1_python.m:52
    Result['Is_SecondOrder'] = Is_SecondOrder
    # GetResults_exact_metric1_python.m:53
    Result['Is_ThirdOrder'] = Is_ThirdOrder
    # GetResults_exact_metric1_python.m:54
    KNNs_ret_file = tmp_prefix + '_KNN_ret.mat'
    # GetResults_exact_metric1_python.m:56
    k_test = 10
    # GetResults_exact_metric1_python.m:57
    if os.path.exists(KNNs_ret_file) and flag_always_compute == 0:
        load(KNNs_ret_file)
    else:
        # print('X_embedding', X_embedding.shape)
        # print('T_unique_embeddings', T_unique_embeddings.shape)
        ret_ind, ret_D = get_exact_cos(X_embedding, T_unique_embeddings, K, nargout=2)
        import sys
        np.set_printoptions(threshold=sys.maxsize)

        # GetResults_exact_metric1_python.m:61
        ret_facts_ind, ret_facts_D = get_exact_cos(T_unique_embeddings, T_unique_embeddings, k_test, nargout=2)
        # GetResults_exact_metric1_python.m:62
        if (flag_always_compute == 0):
            save(KNNs_ret_file, 'ret_ind', 'ret_D', 'ret_facts_ind', 'ret_facts_D')

    metric = 'cos'
    # GetResults_exact_metric1_python.m:70
    ThirdOrderGroups = np.array(list(range(T_unique_embeddings.shape[0])))
    # GetResults_exact_metric1_python.m:75
    T_unique_embeddings_S = T_unique_embeddings_900[:, :300]
    # GetResults_exact_metric1_python.m:77
    #     T_unique_S_embeddings,ia_S,ic_S=unique(T_unique_embeddings_S,'rows',nargout=3)
    T_unique_S_embeddings, ia_S, ic_S = np.unique(T_unique_embeddings_S, return_index=True, return_inverse=True, axis=0)
    # GetResults_exact_metric1_python.m:78
    T_unique_embeddings_SP = T_unique_embeddings_900[:, :600]
    # GetResults_exact_metric1_python.m:80
    #     T_unique_SP_embeddings,ia_PO,ic_PO=unique(T_unique_embeddings_SP,'rows',nargout=3)
    T_unique_SP_embeddings, ia_PO, ic_PO = np.unique(T_unique_embeddings_SP, return_index=True, return_inverse=True,
                                                     axis=0)
    # GetResults_exact_metric1_python.m:81
    FirstOrderGroups = ic_S
    # GetResults_exact_metric1_python.m:83
    SecondOrderGroups = ic_PO
    # GetResults_exact_metric1_python.m:84
    KNNs_kn_file = tmp_prefix + '_KNN_kn.mat'
    # GetResults_exact_metric1_python.m:90
    if (os.path.exists(KNNs_kn_file) and flag_always_compute == 0):
        load(KNNs_kn_file)
    else:
        kn_ind, kn_D = get_exact_cos(T_unique_embeddings, X_embedding, K, nargout=2)
        # GetResults_exact_metric1_python.m:95
        if (flag_always_compute == 0):
            save(KNNs_kn_file, 'kn_ind', 'kn_D')

    Do_retrieval = 1
    # GetResults_exact_metric1_python.m:104
    Do_kn = 1
    # GetResults_exact_metric1_python.m:106
    harsh_ret = 1
    # GetResults_exact_metric1_python.m:107

    # unique_test_ids,ia_test_ids,ic_testids=unique(TestData.tuple_ids,nargout=3)
    unique_test_ids, ia_test_ids, ic_testids = np.unique(TestData['tuple_ids'][0][0], return_index=True,
                                                         return_inverse=True, axis=0)

    # GetResults_exact_metric1_python.m:111
    Result['tuple_ids'] = unique_test_ids
    # GetResults_exact_metric1_python.m:113
    if (Do_kn == 1):
        avgKnowledgeDetRatio_K1 = 0
        # GetResults_exact_metric1_python.m:116
        avgKnowledgeDetRatio_K5 = 0
        # GetResults_exact_metric1_python.m:117
        avgKnowledgeDetRatio_K10 = 0
        # GetResults_exact_metric1_python.m:118
        avgKnowledgeDetRatio_MRR = 0
        # GetResults_exact_metric1_python.m:119
        Result['KnowledgeDetRatios_K10'] = []
        # GetResults_exact_metric1_python.m:121
        Result['KnowledgeDetRatios_K5'] = []
        # GetResults_exact_metric1_python.m:122
        Result['KnowledgeDetRatios_K1'] = []
        # GetResults_exact_metric1_python.m:123
        Result['KnowledgeDetRatios_MRR'] = []
        # GetResults_exact_metric1_python.m:124
        all_im_ind = list(range(X_embedding.shape[0]))
        # GetResults_exact_metric1_python.m:127
        all_tuple_ind = np.array(list(range(T_unique_embeddings.shape[0])))
        # GetResults_exact_metric1_python.m:128
        num_ismember100 = 0
        # GetResults_exact_metric1_python.m:129
        for i in range(X_embedding.shape[0]):
            #              if(mod(i, 1000)==0)
            #                 i
            #              end
            retrieved_tuple_ind_i_100 = kn_ind[i, :]
            # GetResults_exact_metric1_python.m:135
            ids_same_group_bool = ic_imnames == ic_imnames[i]
            # GetResults_exact_metric1_python.m:136
            if (flag_nodup):
                rel_im_ids = [i]
            # GetResults_exact_metric1_python.m:138
            else:
                rel_im_ids = all_im_ind[ids_same_group_bool]
            # GetResults_exact_metric1_python.m:140
            gt_tuple_inds = np.array([])
            # GetResults_exact_metric1_python.m:142
            duplicate_groups = np.array([])
            # GetResults_exact_metric1_python.m:143

            # if i > 3025:
            #     print(gt_tuple_inds, i + 1)
            # print(Is_FirstOrder)
            # print(Is_SecondOrder)
            for j in range(len(rel_im_ids)):
                tuple_ind_i = ic[rel_im_ids[j]]

                # GetResults_exact_metric1_python.m:145
                if Is_FirstOrder[tuple_ind_i]:
                    gt_tuple_inds = np.concatenate(
                        [gt_tuple_inds, all_tuple_ind[np.equal(FirstOrderGroups, FirstOrderGroups[tuple_ind_i])]])
                # GetResults_exact_metric1_python.m:148
                elif Is_SecondOrder[tuple_ind_i]:
                    gt_tuple_inds = np.concatenate(
                        [gt_tuple_inds, all_tuple_ind[np.equal(SecondOrderGroups, SecondOrderGroups[tuple_ind_i])]])
                # GetResults_exact_metric1_python.m:151
                else:
                    gt_tuple_inds = np.concatenate(
                        [gt_tuple_inds, all_tuple_ind[np.equal(ThirdOrderGroups, ThirdOrderGroups[tuple_ind_i])]])


                # GetResults_exact_metric1_python.m:153
                #                 print(np.dot(j, np.ones((1,len(gt_tuple_inds) - len(duplicate_groups)))))
                #                 print(j)
                #                 print(np.ones((1,len(gt_tuple_inds) - len(duplicate_groups))))
                #                 print(duplicate_groups)
                duplicate_groups = np.append(duplicate_groups,
                                             np.dot(j, np.ones((1, len(gt_tuple_inds) - len(duplicate_groups)))))
            # GetResults_exact_metric1_python.m:155
            assert (len(duplicate_groups) == len(gt_tuple_inds))
            # gt_tuple_inds,un_gt_ind=unique(gt_tuple_inds,nargout=2)
            gt_tuple_inds, un_gt_ind, _ = np.unique(gt_tuple_inds, return_index=True, return_inverse=True, axis=0)
            # print('gt_tuple_inds', gt_tuple_inds)



            # GetResults_exact_metric1_python.m:160
            duplicate_groups = duplicate_groups[un_gt_ind]
            # GetResults_exact_metric1_python.m:161
            offset = 0
            # GetResults_exact_metric1_python.m:163
            numAll = 0
            # GetResults_exact_metric1_python.m:164
            numAllOthers = 0
            # GetResults_exact_metric1_python.m:165
            sum_isMember10 = 0
            # GetResults_exact_metric1_python.m:166
            sum_isMember5 = 0
            # GetResults_exact_metric1_python.m:167
            sum_isMember1 = 0
            # GetResults_exact_metric1_python.m:168
            MRR_i = 0
            # GetResults_exact_metric1_python.m:169
            ismemberAll, ismemberAll_ind = ismember(gt_tuple_inds, retrieved_tuple_ind_i_100)
            # print(ismemberAll, ismemberAll_ind)
            # GetResults_exact_metric1_python.m:171
            if (sum(ismemberAll) != 0):
                num_ismember100 = num_ismember100 + 1
            # GetResults_exact_metric1_python.m:174
            ismemberAll_ind[ismemberAll_ind == None] = np.iinfo(np.int32).max

            # GetResults_exact_metric1_python.m:176
            #             ismemberAll_ind_s,ismemberAll_ind_s_ind=sort(ismemberAll_ind,nargout=2)
            ismemberAll_ind_s_ind = np.argsort(ismemberAll_ind)

            # print(ismemberAll_ind_s_ind)
            ismemberAll_ind_s = ismemberAll_ind[ismemberAll_ind_s_ind]

            # GetResults_exact_metric1_python.m:177
            #             duplicate_visited = np.zeros((1,len(duplicate_groups))) == 1
            duplicate_visited = np.zeros((len(duplicate_groups)), dtype=bool)
            # GetResults_exact_metric1_python.m:178

            for j_gt_ind in range(len(gt_tuple_inds)):
                j_gt = ismemberAll_ind_s_ind[j_gt_ind]

                # GetResults_exact_metric1_python.m:180
                if (duplicate_visited[j_gt] == False):
                    if (ismemberAll[j_gt]):
                        # for k=1:len(duplicates_found)
                        duplicate_visited[duplicate_groups == duplicate_groups[j_gt]] = True
                        # GetResults_exact_metric1_python.m:185
                        if (ismemberAll_ind[j_gt] - offset + 1 <= 1 + len(gt_tuple_inds) - 1):
                            sum_isMember1 = sum_isMember1 + 1
                        # GetResults_exact_metric1_python.m:189
                        if (ismemberAll_ind[j_gt] - offset + 1 <= 5 + len(gt_tuple_inds) - 1):
                            sum_isMember5 = sum_isMember5 + 1
                        # GetResults_exact_metric1_python.m:192
                        if (ismemberAll_ind[j_gt] - offset + 1 <= 10 + len(gt_tuple_inds) - 1):
                            sum_isMember10 = sum_isMember10 + 1
                        # GetResults_exact_metric1_python.m:196
                        MRR_i = MRR_i + 1 / (ismemberAll_ind[j_gt] + 1 - offset) # todo: I added one to avoid division by 0
                        # GetResults_exact_metric1_python.m:201
                        numAll = numAll + 1
                        # GetResults_exact_metric1_python.m:202
                        offset = offset + 1
                    # GetResults_exact_metric1_python.m:203
                    numAllOthers = numAllOthers + 1
                # GetResults_exact_metric1_python.m:206
                else:
                    offset = offset + 1
            # GetResults_exact_metric1_python.m:209
            if (numAllOthers > 0):
                detRatio10_i = sum_isMember10 / numAllOthers
                # GetResults_exact_metric1_python.m:215
                detRatio5_i = sum_isMember5 / numAllOthers
                # if detRatio5_i == 1:
                #     debug_counter += 1
                #     print(debug_counter, i+1)

                # GetResults_exact_metric1_python.m:216
                detRatio1_i = sum_isMember1 / numAllOthers
            # GetResults_exact_metric1_python.m:217
            else:
                detRatio10_i = 0
                # GetResults_exact_metric1_python.m:220
                detRatio5_i = 0
                # GetResults_exact_metric1_python.m:221
                detRatio1_i = 0
            # GetResults_exact_metric1_python.m:222
            if (numAll == 0):
                MRR_i = 0
            # GetResults_exact_metric1_python.m:225
            else:
                if (numAllOthers == 0):
                    MRR_i = 0
                # GetResults_exact_metric1_python.m:228
                else:
                    MRR_i = MRR_i / numAllOthers
            # GetResults_exact_metric1_python.m:230
            Result['KnowledgeDetRatios_K10'] += [detRatio10_i]
            # GetResults_exact_metric1_python.m:234
            Result['KnowledgeDetRatios_K5'] += [detRatio5_i]
            # GetResults_exact_metric1_python.m:235
            Result['KnowledgeDetRatios_K1'] += [detRatio1_i]
            # GetResults_exact_metric1_python.m:236
            Result['KnowledgeDetRatios_MRR'] += [MRR_i]
            # GetResults_exact_metric1_python.m:237
            avgKnowledgeDetRatio_K10 = avgKnowledgeDetRatio_K10 + detRatio10_i
            # GetResults_exact_metric1_python.m:239
            avgKnowledgeDetRatio_K5 = avgKnowledgeDetRatio_K5 + detRatio5_i
            # GetResults_exact_metric1_python.m:240
            avgKnowledgeDetRatio_K1 = avgKnowledgeDetRatio_K1 + detRatio1_i
            # GetResults_exact_metric1_python.m:241
            avgKnowledgeDetRatio_MRR = avgKnowledgeDetRatio_MRR + MRR_i
        # GetResults_exact_metric1_python.m:242
        avgKnowledgeDetRatio_K10 = avgKnowledgeDetRatio_K10 / X_embedding.shape[0]
        # GetResults_exact_metric1_python.m:247
        avgKnowledgeDetRatio_K5 = avgKnowledgeDetRatio_K5 / X_embedding.shape[0]
        # GetResults_exact_metric1_python.m:248
        avgKnowledgeDetRatio_K1 = avgKnowledgeDetRatio_K1 / X_embedding.shape[0]
        # GetResults_exact_metric1_python.m:249
        avgKnowledgeDetRatio_MRR = avgKnowledgeDetRatio_MRR / X_embedding.shape[0]
        # GetResults_exact_metric1_python.m:250
        Result['KnowledgeMeanDetRatio_K10'] = avgKnowledgeDetRatio_K10
        # GetResults_exact_metric1_python.m:253
        Result['KnowledgeMeanDetRatio_K5'] = avgKnowledgeDetRatio_K5
        # GetResults_exact_metric1_python.m:254
        Result['KnowledgeMeanDetRatio_K1'] = avgKnowledgeDetRatio_K1
        # GetResults_exact_metric1_python.m:255
        Result['KnowledgeMeanDetRatio_MRR'] = avgKnowledgeDetRatio_MRR
    # GetResults_exact_metric1_python.m:256

    x = 1
    # GetResults_exact_metric1_python.m:260
    if (Do_retrieval == 1):
        AUCs = np.zeros((T_unique_embeddings.shape[0]))
        # GetResults_exact_metric1_python.m:264
        APs = np.zeros((T_unique_embeddings.shape[0]))
        # GetResults_exact_metric1_python.m:266
        AP10s = np.zeros((T_unique_embeddings.shape[0]))
        # GetResults_exact_metric1_python.m:267
        AP100s = np.zeros((T_unique_embeddings.shape[0]))
        # GetResults_exact_metric1_python.m:268
        Top1Accs = np.zeros((T_unique_embeddings.shape[0]))
        # GetResults_exact_metric1_python.m:270
        Top5Accs = np.zeros((T_unique_embeddings.shape[0]))
        # GetResults_exact_metric1_python.m:272
        Top10Accs = np.zeros((T_unique_embeddings.shape[0]))
        # GetResults_exact_metric1_python.m:274
        Top20Accs = np.zeros((T_unique_embeddings.shape[0]))
        # GetResults_exact_metric1_python.m:276
        Top1APs = np.zeros((T_unique_embeddings.shape[0]))
        # GetResults_exact_metric1_python.m:278
        Top5APs = np.zeros((T_unique_embeddings.shape[0]))
        # GetResults_exact_metric1_python.m:280
        Top10APs = np.zeros((T_unique_embeddings.shape[0]))
        # GetResults_exact_metric1_python.m:282
        Top20APs = np.zeros((T_unique_embeddings.shape[0]))
        # GetResults_exact_metric1_python.m:284
        for i in range(T_unique_embeddings.shape[0]):
            T_embedding_i = T_embedding[ia[i], :]
            # GetResults_exact_metric1_python.m:295
            # X_embedding_scores_i = GetX_Scores_ANN(dbs, X_embedding,T_embedding_i, metric, 100);
            # X_embedding_scores_i = -1000*ones(1, size(X_embedding, 1));
            # X_embedding_scores_i(ret_ind(i,:)) = 1- pdist2(X_embedding(ret_ind(i,:),:), T_embedding_i, 'cos');
            #     unique_tuple_ids_i = unique(TestData.tuple_ids(ic==i));
            # if(size(unique_tuple_ids_i,1)>1)
            #    unique_tuple_ids_i = unique_tuple_ids_i';
            # end
            # X_embedding_gt_i_ret_ind  = np.zeros(1, size(ret_ind,2))==1;
            # for k=1:len(unique_tuple_ids_i)
            #     X_embedding_gt_i_ret_ind = X_embedding_gt_i_ret_ind|(TestData.tuple_ids(ret_ind(i,:))==unique_tuple_ids_i(k));
            # end
            # X_embedding_gt_i_ret_ind  = np.zeros(1, size(ret_ind,2))==1;
            debug = ''
            if (Is_ThirdOrder[i]):
                debug = 'third'
                # X_embedding_scores_i_ret_ind = 1 - pdist2(X_embedding(ret_ind(i, arange()), arange()), T_embedding_i, 'cos')
                X_embedding_scores_i_ret_ind = distance.cdist(X_embedding[ret_ind[i, :], :], np.expand_dims(T_embedding_i, axis=0), 'cosine')
                # GetResults_exact_metric1_python.m:314
                tuple_indeces = ic[ret_ind[i, :]]
                # GetResults_exact_metric1_python.m:315
                tuple_groups = ThirdOrderGroups[tuple_indeces]
                # GetResults_exact_metric1_python.m:316
                X_embedding_gt_i_ret_ind = np.equal(tuple_groups, ThirdOrderGroups[i])
                # GetResults_exact_metric1_python.m:317
                if (harsh_ret != 1):
                    for j in range(ret_facts_ind_SP.shape[1]):
                        X_embedding_gt_i_ret_ind = np.logical_or(X_embedding_gt_i_ret_ind, (
                                np.equal(tuple_groups, ThirdOrderGroups[ret_facts_ind[i, j]])))
            # GetResults_exact_metric1_python.m:320
            elif (Is_SecondOrder[i]):
                debug = 'second'

                # X_embedding_scores_i_ret_ind = 1 - pdist2(X_embedding(ret_ind(i, arange()), arange()), T_embedding_i, 'cos')
                X_embedding_scores_i_ret_ind = 1 - distance.cdist(X_embedding[ret_ind[i, :], :], np.expand_dims(T_embedding_i, axis=0), 'cosine')

                # GetResults_exact_metric1_python.m:324
                tuple_indeces = ic[ret_ind[i, :]]
                # GetResults_exact_metric1_python.m:325
                tuple_groups = SecondOrderGroups[tuple_indeces]
                # GetResults_exact_metric1_python.m:326
                X_embedding_gt_i_ret_ind = np.equal(tuple_groups, SecondOrderGroups[i])
                # GetResults_exact_metric1_python.m:328
                if harsh_ret != 1:
                    for j in range(ret_facts_ind_SP.shape[1]):
                        X_embedding_gt_i_ret_ind = np.logical_or(X_embedding_gt_i_ret_ind, (
                                tuple_groups == SecondOrderGroups[ret_facts_ind_SP[i, j]]))
            # GetResults_exact_metric1_python.m:331
            else:
                debug = 'first'

                # X_embedding_scores_i_ret_ind = 1 - pdist2(X_embedding[ret_ind[i, :], :], T_embedding_i, 'cos')
                X_embedding_scores_i_ret_ind = 1 - distance.cdist(X_embedding[ret_ind[i, :], :], np.expand_dims(T_embedding_i, axis=0), 'cosine')
                # GetResults_exact_metric1_python.m:335
                tuple_indeces = ic[ret_ind[i, :]]
                # GetResults_exact_metric1_python.m:336
                tuple_groups = FirstOrderGroups[tuple_indeces]
                # GetResults_exact_metric1_python.m:337
                # print(ret_ind[i, :] + 1)
                X_embedding_gt_i_ret_ind = np.equal(tuple_groups, FirstOrderGroups[i])
                # GetResults_exact_metric1_python.m:338
                if (harsh_ret != 1):
                    for j in range(ret_facts_ind_S.shape[1]):
                        X_embedding_gt_i_ret_ind = np.logical_or(X_embedding_gt_i_ret_ind, (tuple_groups == FirstOrderGroups[ret_facts_ind_S[i, j]]))
            # GetResults_exact_metric1_python.m:341
            if (np.sum(X_embedding_gt_i_ret_ind) > 0):
                x = 1
            # GetResults_exact_metric1_python.m:346
            # X_embedding_gt_i_ret_ind_pos_ind = find(X_embedding_gt_i_ret_ind)
            X_embedding_gt_i_ret_ind_pos_ind = np.nonzero(X_embedding_gt_i_ret_ind)[0]

            # GetResults_exact_metric1_python.m:349
            # if (np.logical_not(isempty(X_embedding_gt_i_ret_ind_pos_ind))):
            if (np.logical_not(X_embedding_gt_i_ret_ind_pos_ind.size == 0)):
                Top1Acc_i = sum(X_embedding_gt_i_ret_ind_pos_ind < len(X_embedding_gt_i_ret_ind_pos_ind)) / len(
                    X_embedding_gt_i_ret_ind_pos_ind)
                # GetResults_exact_metric1_python.m:351
                Top5Acc_i = sum(X_embedding_gt_i_ret_ind_pos_ind < len(X_embedding_gt_i_ret_ind_pos_ind) + 4) / len(
                    X_embedding_gt_i_ret_ind_pos_ind)
                # GetResults_exact_metric1_python.m:352
                Top10Acc_i = sum(X_embedding_gt_i_ret_ind_pos_ind < len(X_embedding_gt_i_ret_ind_pos_ind) + 9) / len(
                    X_embedding_gt_i_ret_ind_pos_ind)
                # GetResults_exact_metric1_python.m:353
                Top20Acc_i = sum(X_embedding_gt_i_ret_ind_pos_ind < len(X_embedding_gt_i_ret_ind_pos_ind) + 19) / len(
                    X_embedding_gt_i_ret_ind_pos_ind)
                # GetResults_exact_metric1_python.m:354
                Top1AP_i = sum(X_embedding_gt_i_ret_ind_pos_ind < 1) / 1
                # GetResults_exact_metric1_python.m:356
                Top5AP_i = sum(X_embedding_gt_i_ret_ind_pos_ind < 5) / 5
                # GetResults_exact_metric1_python.m:357
                Top10AP_i = sum(X_embedding_gt_i_ret_ind_pos_ind < 10) / 10
                # GetResults_exact_metric1_python.m:358
                Top20AP_i = sum(X_embedding_gt_i_ret_ind_pos_ind < 20) / 20
            # GetResults_exact_metric1_python.m:359
            else:
                Top1Acc_i = 0
                # GetResults_exact_metric1_python.m:362
                Top5Acc_i = 0
                # GetResults_exact_metric1_python.m:363
                Top10Acc_i = 0
                # GetResults_exact_metric1_python.m:364
                Top20Acc_i = 0
                # GetResults_exact_metric1_python.m:365
                Top1AP_i = 0
                # GetResults_exact_metric1_python.m:367
                Top5AP_i = 0
                # GetResults_exact_metric1_python.m:368
                Top10AP_i = 0
                # GetResults_exact_metric1_python.m:369
                Top20AP_i = 0
            # GetResults_exact_metric1_python.m:370
            #         if(strcmp(metric, 'dot'))
            #
            #             X_embedding_scores_i = X_embedding*T_embedding_i';
            #         elseif(strcmp(metric, 'cos'))
            #             X_embedding_scores_i = 1- pdist2(X_embedding, T_embedding_i, 'cos');
            #         elseif(strcmp(metric, 'euc'))
            #             eucDist = pdist2(X_embedding, T_embedding_i, 'euclidean');
            #             sigma=  median(eucDist);
            #             X_embedding_scores_i = exp(-eucDist.^2/(2*sigma^2 ) );
            #         else
            #             error('incorrect metric');
            #         end
            X_embedding_gt_i_ret_ind = np.dot(X_embedding_gt_i_ret_ind, 2) - 1
            # GetResults_exact_metric1_python.m:396
            # [MAP_v2_i] = calcAP_v2(X_embedding_scores_i_ret_ind, X_embedding_gt_i_ret_ind);
            try:
                pass
                # AUC_i, xs, ys = colAUC(X_embedding_scores_i_ret_ind, X_embedding_gt_i_ret_ind, 'ROC', nargout=3)
                # AUC_i = sklearn.metrics.roc_auc_score(X_embedding_scores_i_ret_ind, X_embedding_gt_i_ret_ind)
            # GetResults_exact_metric1_python.m:401
            finally:
                pass
            AP10s[i], _ = calcAP(X_embedding_scores_i_ret_ind, X_embedding_gt_i_ret_ind, 10) # todo: continue from here
            # GetResults_exact_metric1_python.m:405
            AP100s[i], _  = calcAP(X_embedding_scores_i_ret_ind, X_embedding_gt_i_ret_ind, 100)
            # GetResults_exact_metric1_python.m:406
            # AUCs[i] = AUC_i
            # GetResults_exact_metric1_python.m:407
            APs[i], _  = calcAP(X_embedding_scores_i_ret_ind, X_embedding_gt_i_ret_ind, 0)
            # import sklearn
            # APs[i]  = sklearn.metrics.average_precision_score(X_embedding_scores_i_ret_ind, X_embedding_gt_i_ret_ind)
            # GetResults_exact_metric1_python.m:408
            Top1Accs[i] = Top1Acc_i
            # GetResults_exact_metric1_python.m:410
            Top5Accs[i] = Top5Acc_i
            # GetResults_exact_metric1_python.m:411
            Top10Accs[i] = Top10Acc_i
            # GetResults_exact_metric1_python.m:412
            Top20Accs[i] = Top20Acc_i
            # GetResults_exact_metric1_python.m:413
            Top1APs[i] = Top1AP_i
            # GetResults_exact_metric1_python.m:415
            Top5APs[i] = Top5AP_i
            # GetResults_exact_metric1_python.m:416
            Top10APs[i] = Top10AP_i
            # GetResults_exact_metric1_python.m:417
            Top20APs[i] = Top20AP_i
        # GetResults_exact_metric1_python.m:418
        #               Top1AP_i= sum(X_embedding_gt_i_ret_ind_pos_ind<=1)/1;
        #              Top5AP_i= sum(X_embedding_gt_i_ret_ind_pos_ind<=5)/5;
        #              Top10AP_i=sum(X_embedding_gt_i_ret_ind_pos_ind<=10)/10;
        #              Top20AP_i=sum(X_embedding_gt_i_ret_ind_pos_ind<=20)/20;
        Result['AUCs'] = AUCs
        # GetResults_exact_metric1_python.m:425
        Result['APs'] = APs
        # GetResults_exact_metric1_python.m:426
        Result['AP10s'] = AP10s
        # GetResults_exact_metric1_python.m:428
        Result['AP100s'] = AP100s
        # GetResults_exact_metric1_python.m:429
        Result['mAP'] = np.mean(APs)
        # GetResults_exact_metric1_python.m:433
        Result['mAP10s'] = np.mean(AP10s)
        # GetResults_exact_metric1_python.m:435
        Result['mAP100s'] = np.mean(AP100s)
        # GetResults_exact_metric1_python.m:436
        Result['Top1Accs'] = Top1Accs
        # GetResults_exact_metric1_python.m:438
        Result['Top5Accs'] = Top5Accs
        # GetResults_exact_metric1_python.m:439
        Result['Top10Accs'] = Top10Accs
        # GetResults_exact_metric1_python.m:440
        Result['Top20Accs'] = Top20Accs
        # GetResults_exact_metric1_python.m:441
        Result['mTop1Acc'] = np.mean(Top1Accs)
        # GetResults_exact_metric1_python.m:443
        Result['mTop5Acc'] = np.mean(Top5Accs)
        # GetResults_exact_metric1_python.m:444
        Result['mTop10Acc'] = np.mean(Top10Accs)
        # GetResults_exact_metric1_python.m:445
        Result['mTop20Acc'] = np.mean(Top20Accs)
        # GetResults_exact_metric1_python.m:446
        #                      Top1APs(i) = Top1AP_i;
        #              Top5APs(i) = Top5AP_i;
        #              Top10APs(i) = Top10AP_i;
        #              Top20APs(i) = Top20AP_i;
        Result['Top10APs'] = Top1APs
        # GetResults_exact_metric1_python.m:453
        Result['Top5APs'] = Top5APs
        # GetResults_exact_metric1_python.m:454
        Result['Top10APs'] = Top10APs
        # GetResults_exact_metric1_python.m:455
        Result['Top20APs'] = Top20APs
        # GetResults_exact_metric1_python.m:456
        Result['mTop10AP'] = np.mean(Top10APs)
        # GetResults_exact_metric1_python.m:458
        Result['mTop5AP'] = np.mean(Top5APs)
        # GetResults_exact_metric1_python.m:459
        Result['mTop10AP'] = np.mean(Top10APs)
        # GetResults_exact_metric1_python.m:460
        Result['mTop20AP'] = np.mean(Top20APs)

    return Result

# GetResults_exact_metric1_python.m:461


def get_val_array(map_object=None, key_set=None, *args, **kwargs):
    varargin = get_val_array.varargin
    nargin = get_val_array.nargin

    valset = np.zeros(len(key_set))
    # GetResults_exact_metric1_python.m:467
    for i in range(len(key_set)):
        valset[i] = map_object(key_set(i))
# GetResults_exact_metric1_python.m:469
