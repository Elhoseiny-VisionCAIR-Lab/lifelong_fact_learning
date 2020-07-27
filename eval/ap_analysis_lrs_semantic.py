import pandas as pd
import matplotlib.pyplot as plt
import csv
import argparse
import os

import pickle
import numpy as np
import json

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

import scipy.io
from get_exact_cos import get_exact_cos


def evaluate_each(t_embedding, x_embedding, gt, k):
    gt = np.array(gt.values.tolist())
    un, ia, ic = np.unique(gt, return_index=True, return_inverse=True)
    classes = gt[ia]
    nan_idx = np.where(classes == 'nan')[0][0]
    classes = np.delete(classes, nan_idx)
    # print(len(classes))

    # t_unique_embeddings_900, ia, ic = np.unique(t_embedding, return_index=True, return_inverse=True, axis=0)
    t_unique_embeddings = t_embedding[ia, :]
    t_unique_embeddings = np.delete(t_unique_embeddings, nan_idx, axis=0)


    # print(t_embedding)

    # print(t_unique_embeddings.shape)
    # print(t_embedding.shape)
    # print(t_unique_embeddings.shape)
    # ret_ind, ret_D = get_exact_cos(x_embedding, t_unique_embeddings, K, nargout=2)

    # print(t_unique_embeddings.shape, x_embedding.shape)
    ret_ind, ret_D = get_exact_cos(t_unique_embeddings, x_embedding, k, nargout=2)
    # print('x_embedding', x_embedding.shape)
    # print('t_embedding', t_embedding.shape)
    # print('t_unique_embeddings', t_unique_embeddings.shape)
    # print('ret_ind', ret_ind)
    # print('ret_ind', np.max(ret_ind))
    # print('ret_ind', ret_ind[0, :])

    k = min(k, len(classes))
    predictions = classes[ret_ind]
    distances = np.sort(ret_D, axis=1)
    similarity = 1/distances
    scores = np.exp(similarity)/np.sum(np.exp(similarity), axis=1, keepdims=True)

    topk = pd.DataFrame(columns=['image_id', 'det_id', 'gt', 'det', 'score'])

    for i in range(predictions.shape[1]):
        for m in range(k):
            det = predictions[i, m]
            gt_i = gt[i]
            score = scores[i, m]
            topk = topk.append({'image_id': i, 'det_id': m, 'gt': gt_i, 'det': det, 'score': score}, ignore_index=True)
    return topk
    # with open(csv_path, 'w', newline='') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #
    #     spamwriter.writerow(['image_id',
    #                          'det_id',
    #
    #                          'gt',
    #                          'det',
    #                          'score'])
    #
    #     for i in range(predictions.shape[1]):
    #         for m in range(k):
    #             # print(predictions.shape, i, m)
    #             det = predictions[i, m]
    #             gt_i = gt[i]
    #             score = scores[i, m]
    #             spamwriter.writerow(
    #                 [i,
    #                  m,
    #
    #                  gt_i,
    #                  det,
    #                  score])

    # print(facts.loc[, :])
    # print(ret_D)


def prepare_df_prd(df):
    print('Reading..')
    w2v_gn_obj = pd.read_pickle('./eval_files/gt_words/similarity_matrices/prd_sim_w2v.pkl')
    # w2v_gn_prd = pd.read_pickle('./eval_files/gt_words/similarity_matrices/prd_sim_w2v.pkl')
    # w2v_relco_obj = pd.read_pickle('./eval_files/gt_words/similarity_matrices/obj_sim_w2v_relco.pkl')
    # w2v_relco_prd = pd.read_pickle('./eval_files/gt_words/similarity_matrices/prd_sim_w2v_relco.pkl')

    # print(df['det'].unique())
    # print(df['gt'].unique())
    # print(lch.columns)
    df['w2v'] = w2v_gn_obj.lookup(df['gt'], df['det'])
    # df['w2v_relco'] = w2v_relco_obj.lookup(df['gt'], df['det'])
    df['exact_match'] = df['gt'] == df['det']
    df['exact_match'] = df['exact_match'].astype(int)

    return df


def prepare_df(df):
    print('Reading..')
    lch = pd.read_pickle('./eval_files/gt_words/similarity_matrices/obj_sim_lch.pkl')
    wup = pd.read_pickle('./eval_files/gt_words/similarity_matrices/obj_sim_wup.pkl')
    res = pd.read_pickle('./eval_files/gt_words/similarity_matrices/obj_sim_res.pkl')
    jcn = pd.read_pickle('./eval_files/gt_words/similarity_matrices/obj_sim_jcn.pkl')
    lin = pd.read_pickle('./eval_files/gt_words/similarity_matrices/obj_sim_lin.pkl')
    path = pd.read_pickle('./eval_files/gt_words/similarity_matrices/obj_sim_path.pkl')
    w2v_gn_obj = pd.read_pickle('./eval_files/gt_words/similarity_matrices/obj_sim_w2v.pkl')
    # w2v_gn_prd = pd.read_pickle('./eval_files/gt_words/similarity_matrices/prd_sim_w2v.pkl')
    # w2v_relco_obj = pd.read_pickle('./eval_files/gt_words/similarity_matrices/obj_sim_w2v_relco.pkl')
    # w2v_relco_prd = pd.read_pickle('./eval_files/gt_words/similarity_matrices/prd_sim_w2v_relco.pkl')

    # print(df['det'].unique())
    # print(df['gt'].unique())
    # print(lch.columns)
    df['lch'] = lch.lookup(df['gt'], df['det'])
    df['wup'] = wup.lookup(df['gt'], df['det'])
    df['res'] = res.lookup(df['gt'], df['det'])
    df['jcn'] = jcn.lookup(df['gt'], df['det'])
    df['lin'] = lin.lookup(df['gt'], df['det'])
    df['path'] = path.lookup(df['gt'], df['det'])
    df['w2v'] = w2v_gn_obj.lookup(df['gt'], df['det'])
    # df['w2v_relco'] = w2v_relco_obj.lookup(df['gt'], df['det'])
    df['exact_match'] = df['gt'] == df['det']
    df['exact_match'] = df['exact_match'].astype(int)


    # df['rel_lch'] = lch.lookup(df['gt_rel'], df['det_rel'])

    # print(df.head)
    return df


def do_analysis(T1_threshold, df):
    # df = pd.read_csv(csv_path)
    df = df.dropna().reset_index()
    df = prepare_df(df)

    length = len(df['image_id'])

    # print('df', length)
    metrics = np.concatenate((df['lch'].to_numpy().reshape((length, 1)),
                              df['wup'].to_numpy().reshape((length, 1)),
                              df['res'].to_numpy().reshape((length, 1)),
                              df['jcn'].to_numpy().reshape((length, 1)),
                              df['lin'].to_numpy().reshape((length, 1)),
                              df['path'].to_numpy().reshape((length, 1)),
                              df['exact_match'].to_numpy().reshape((length, 1)),
                              df['w2v'].to_numpy().reshape((length, 1))), axis=1)
                              # df['w2v_relco'].to_numpy().reshape((length, 1))), axis=1) # shape: (num_examples, num_metrics)

    # metric_names = ['lch_similarity', 'wup_similarity', 'res_similarity', 'jcn_similarity', 'lin_similarity',
    #                 'path_similarity', 'exact_match', 'word2vec_GNews', 'word2vec_relco']
    metric_names = ['lch_similarity', 'wup_similarity', 'res_similarity', 'jcn_similarity', 'lin_similarity',
                    'path_similarity', 'exact_match', 'word2vec_GNews']

    mean_metric = np.zeros((metrics.shape[1]))      # (num_metric,)
    min_metric = np.zeros((metrics.shape[1]))       # (num_metric,)

    max_metric = np.zeros((metrics.shape[1]))       # (num_metric,)

    median_metric = np.zeros((metrics.shape[1]))    # (num_metric,)

    std_metric = np.zeros((metrics.shape[1]))       # (num_metric,)

    for i in range(metrics.shape[1]):
        mean_metric[i] = np.mean(metrics[:, i])     # get mean, std, min, max, and median of each metric
        std_metric[i] = np.std(metrics[:, i])
        min_metric[i] = np.min(metrics[:, i])
        max_metric[i] = np.max(metrics[:, i])
        median_metric[i] = np.median(metrics[:, i])

    ranking_size = 50

    # mean_metric = np.zeros((metrics.shape[1]))

    judges = np.zeros_like(metrics)

    for i in range(metrics.shape[1]):
        for k in range(int(metrics.shape[0] / ranking_size)):
            indices_ik = np.argsort(metrics[k * ranking_size:k * ranking_size + ranking_size, i])[::-1]
            judges[k * ranking_size + indices_ik[0:T1_threshold], i] = 1

    scores_system = np.zeros_like(metrics[:, 0])        # shape: (num_examples, 1)
    scores_rank_system = np.zeros_like(metrics[:, 0])   # shape: (num_examples, 1)
    curr_index = 0

    detections = [{'scores': []} for _ in range(np.max(df['image_id']) + 1)]

    for i in range(len(df)):
        detections[df['image_id'][i]]['scores'].append(df['score'][i])

    for i in range(len(detections)):
        # scores_i = img_scores[i]
        sorted_scores = detections[i]['scores']
        scores_i = sorted_scores[:ranking_size]
        scores_system[curr_index:curr_index + len(scores_i)] = scores_i[:]
        scores_rank_system[curr_index:curr_index + len(scores_i)] = np.flip(np.arange(ranking_size))
        curr_index = curr_index + len(scores_i)

    mask_top_1 = scores_rank_system >= ranking_size - 1
    mask_top_5 = scores_rank_system >= ranking_size - 5
    mask_top_10 = scores_rank_system >= ranking_size - 10
    # mask_top_20 = scores_rank_system >= 230
    # mask_top_50 = scores_rank_system >= 200

    judges_exact = judges[:, 6]
    # print(judges)
    # exit()
    std_accuracy = average_precision_score(judges_exact[mask_top_1], scores_rank_system[mask_top_1]) # standard accuracy
    print('accuracy', metric_names[6], std_accuracy)

    Table_score_summary = np.zeros((metrics.shape[1], 3))

    Table_rank_summary = np.zeros((metrics.shape[1], 3))

    ap_scores_top5 = []
    ap_scores_top10 = []
    ap_rank_top5 = []
    ap_rank_top10 = []

    for i in range(metrics.shape[1]):
        # metric_i = metrics[:,i]
        judges_i = judges[:, i]
        judges_i[judges_exact == 1] = 1

        average_precision_i_score = average_precision_score(judges_i, scores_system)
        average_precision_i_rank = average_precision_score(judges_i, scores_rank_system)

        average_precision_i_score_1 = average_precision_score(judges_i[mask_top_1], scores_system[mask_top_1])
        average_precision_i_rank_1 = average_precision_score(judges_i[mask_top_1], scores_rank_system[mask_top_1])

        average_precision_i_score_5 = average_precision_score(judges_i[mask_top_5], scores_system[mask_top_5])
        average_precision_i_rank_5 = average_precision_score(judges_i[mask_top_5], scores_rank_system[mask_top_5])

        average_precision_i_score_10 = average_precision_score(judges_i[mask_top_10], scores_system[mask_top_10])
        average_precision_i_rank_10 = average_precision_score(judges_i[mask_top_10], scores_rank_system[mask_top_10])

        # average_precision_i_score_20 = average_precision_score(judges_i[mask_top_20], scores_system[mask_top_20])
        # average_precision_i_rank_20 = average_precision_score(judges_i[mask_top_20], scores_rank_system[mask_top_20])
        #
        # average_precision_i_score_50 = average_precision_score(judges_i[mask_top_50], scores_system[mask_top_50])
        # average_precision_i_rank_50 = average_precision_score(judges_i[mask_top_50], scores_rank_system[mask_top_50])

        # Table_score_summary[i, :] = [average_precision_i_score_1, average_precision_i_score_5,
        #                              average_precision_i_score_10, average_precision_i_score_20,
        #                              average_precision_i_score_50, average_precision_i_score]
        Table_score_summary[i, :] = [average_precision_i_score_1, average_precision_i_score_5,
                                     average_precision_i_score]
        Table_rank_summary[i, :] = [average_precision_i_rank_1, average_precision_i_rank_5, average_precision_i_rank]

        ap_scores_top5.append('{:0.2f}%'.format(average_precision_i_score_5 * 100))
        ap_scores_top10.append('{:0.2f}%'.format(average_precision_i_score_10 * 100))
        ap_rank_top5.append('{:0.2f}%'.format(average_precision_i_rank_5 * 100))
        ap_rank_top10.append('{:0.2f}%'.format(average_precision_i_rank_10 * 100))

        # print(metric_names[i], 'AP by score = ', '{:0.2f}%'.format(average_precision_i_score * 100),
        #       'AP by rank = ', '{:0.2f}%'.format(average_precision_i_rank * 100))
        #
        # print(metric_names[i], 'AP by score at Top 1 = ', '{:0.2f}%'.format(average_precision_i_score_1 * 100),
        #       'AP by rank at Top 1 = ', '{:0.2f}%'.format(average_precision_i_rank_1 * 100))
        #
        # print(metric_names[i], 'AP by score at Top 5 = ', '{:0.2f}%'.format(average_precision_i_score_5 * 100),
        #       'AP by rank at Top 5 = ', '{:0.2f}%'.format(average_precision_i_rank_5 * 100))
        #
        # print(metric_names[i], 'AP by score at Top 10 = ', '{:0.2f}%'.format(average_precision_i_score_10 * 100),
        #       'AP by rank  at Top 10 = ', '{:0.2f}%'.format(average_precision_i_rank_10 * 100))
        #
        # print(metric_names[i], prediction_type, ', AP by score at Top 20 = ', '{:0.2f}%'.format(average_precision_i_score_20),
        #       'AP by rank  at Top 20 = ', '{:0.2f}%'.format(average_precision_i_rank_20))
        #
        # print(metric_names[i], prediction_type, ', AP by score at Top 50 = ', '{:0.2f}%'.format(average_precision_i_score_50),
        #       'AP by rank  at Top 50 = ', '{:0.2f}%'.format(average_precision_i_rank_50))
        #

    print('Metric_type\tLCH\tWUP\tRES\tJCN\tLIN\tPATH\tExact\tW2V_GN:')
    print('AP scores top 5\t', *ap_scores_top5, sep='\t')
    print('AP scores top 10\t', *ap_scores_top10, sep='\t')
    print('AP ranks top 5\t', *ap_rank_top5, sep='\t')
    print('AP ranks top 10\t', *ap_rank_top10, sep='\t')

    Table_score_summary = np.transpose(Table_score_summary)

    Table_rank_summary = np.transpose(Table_rank_summary)

    # table_score_fname = outfile_prefix + '_scores_{}.csv'.format(prediction_type)
    # table_rank_fname = outfile_prefix + '_rank_{}.csv'.format(prediction_type)

    # np.savetxt(table_score_fname, Table_score_summary, delimiter=",", header=','.join(metric_names), comments='')
    # np.savetxt(table_rank_fname, Table_rank_summary, delimiter=",", header=','.join(metric_names), comments='')


def do_analysis_prd(T1_threshold, df):
    # df = pd.read_csv(csv_path)
    df = df.dropna().reset_index()
    df = prepare_df_prd(df)

    length = len(df['image_id'])

    # print('df', length)
    metrics = np.concatenate((df['exact_match'].to_numpy().reshape((length, 1)),
                              df['w2v'].to_numpy().reshape((length, 1))), axis=1)
                              # df['w2v_relco'].to_numpy().reshape((length, 1))), axis=1) # shape: (num_examples, num_metrics)

    # metric_names = ['lch_similarity', 'wup_similarity', 'res_similarity', 'jcn_similarity', 'lin_similarity',
    #                 'path_similarity', 'exact_match', 'word2vec_GNews', 'word2vec_relco']
    metric_names = ['exact_match', 'word2vec_GNews']

    mean_metric = np.zeros((metrics.shape[1]))      # (num_metric,)
    min_metric = np.zeros((metrics.shape[1]))       # (num_metric,)

    max_metric = np.zeros((metrics.shape[1]))       # (num_metric,)

    median_metric = np.zeros((metrics.shape[1]))    # (num_metric,)

    std_metric = np.zeros((metrics.shape[1]))       # (num_metric,)

    for i in range(metrics.shape[1]):
        mean_metric[i] = np.mean(metrics[:, i])     # get mean, std, min, max, and median of each metric
        std_metric[i] = np.std(metrics[:, i])
        min_metric[i] = np.min(metrics[:, i])
        max_metric[i] = np.max(metrics[:, i])
        median_metric[i] = np.median(metrics[:, i])

    ranking_size = 50

    # mean_metric = np.zeros((metrics.shape[1]))

    judges = np.zeros_like(metrics)

    for i in range(metrics.shape[1]):
        for k in range(int(metrics.shape[0] / ranking_size)):
            indices_ik = np.argsort(metrics[k * ranking_size:k * ranking_size + ranking_size, i])[::-1]
            judges[k * ranking_size + indices_ik[0:T1_threshold], i] = 1

    scores_system = np.zeros_like(metrics[:, 0])        # shape: (num_examples, 1)
    scores_rank_system = np.zeros_like(metrics[:, 0])   # shape: (num_examples, 1)
    curr_index = 0

    detections = [{'scores': []} for _ in range(np.max(df['image_id']) + 1)]

    for i in range(len(df)):
        detections[df['image_id'][i]]['scores'].append(df['score'][i])

    for i in range(len(detections)):
        # scores_i = img_scores[i]
        sorted_scores = detections[i]['scores']
        scores_i = sorted_scores[:ranking_size]
        scores_system[curr_index:curr_index + len(scores_i)] = scores_i[:]
        scores_rank_system[curr_index:curr_index + len(scores_i)] = np.flip(np.arange(ranking_size))
        curr_index = curr_index + len(scores_i)

    mask_top_1 = scores_rank_system >= ranking_size - 1
    mask_top_5 = scores_rank_system >= ranking_size - 5
    mask_top_10 = scores_rank_system >= ranking_size - 10
    # mask_top_20 = scores_rank_system >= 230
    # mask_top_50 = scores_rank_system >= 200

    judges_exact = judges[:, 0]
    # print(judges)
    # exit()
    std_accuracy = average_precision_score(judges_exact[mask_top_1], scores_rank_system[mask_top_1]) # standard accuracy
    print('accuracy', metric_names[0], std_accuracy)

    Table_score_summary = np.zeros((metrics.shape[1], 3))

    Table_rank_summary = np.zeros((metrics.shape[1], 3))

    ap_scores_top5 = []
    ap_scores_top10 = []
    ap_rank_top5 = []
    ap_rank_top10 = []

    for i in range(metrics.shape[1]):
        # metric_i = metrics[:,i]
        judges_i = judges[:, i]
        judges_i[judges_exact == 1] = 1

        average_precision_i_score = average_precision_score(judges_i, scores_system)
        average_precision_i_rank = average_precision_score(judges_i, scores_rank_system)

        average_precision_i_score_1 = average_precision_score(judges_i[mask_top_1], scores_system[mask_top_1])
        average_precision_i_rank_1 = average_precision_score(judges_i[mask_top_1], scores_rank_system[mask_top_1])

        average_precision_i_score_5 = average_precision_score(judges_i[mask_top_5], scores_system[mask_top_5])
        average_precision_i_rank_5 = average_precision_score(judges_i[mask_top_5], scores_rank_system[mask_top_5])

        average_precision_i_score_10 = average_precision_score(judges_i[mask_top_10], scores_system[mask_top_10])
        average_precision_i_rank_10 = average_precision_score(judges_i[mask_top_10], scores_rank_system[mask_top_10])

        # average_precision_i_score_20 = average_precision_score(judges_i[mask_top_20], scores_system[mask_top_20])
        # average_precision_i_rank_20 = average_precision_score(judges_i[mask_top_20], scores_rank_system[mask_top_20])
        #
        # average_precision_i_score_50 = average_precision_score(judges_i[mask_top_50], scores_system[mask_top_50])
        # average_precision_i_rank_50 = average_precision_score(judges_i[mask_top_50], scores_rank_system[mask_top_50])

        # Table_score_summary[i, :] = [average_precision_i_score_1, average_precision_i_score_5,
        #                              average_precision_i_score_10, average_precision_i_score_20,
        #                              average_precision_i_score_50, average_precision_i_score]
        Table_score_summary[i, :] = [average_precision_i_score_1, average_precision_i_score_5,
                                     average_precision_i_score]
        Table_rank_summary[i, :] = [average_precision_i_rank_1, average_precision_i_rank_5, average_precision_i_rank]

        # print(metric_names[i], 'AP by score = ', '{:0.2f}%'.format(average_precision_i_score * 100),
        #       'AP by rank = ', '{:0.2f}%'.format(average_precision_i_rank * 100))

        # print(metric_names[i], 'AP by score at Top 1 = ', '{:0.2f}%'.format(average_precision_i_score_1 * 100),
        #       'AP by rank at Top 1 = ', '{:0.2f}%'.format(average_precision_i_rank_1 * 100))
        ap_scores_top5.append('{:0.2f}%'.format(average_precision_i_score_5 * 100))
        ap_scores_top10.append('{:0.2f}%'.format(average_precision_i_score_10 * 100))
        ap_rank_top5.append('{:0.2f}%'.format(average_precision_i_rank_5 * 100))
        ap_rank_top10.append('{:0.2f}%'.format(average_precision_i_rank_10 * 100))

        # print(metric_names[i], 'AP by score at Top 5 = ', '{:0.2f}%'.format(average_precision_i_score_5 * 100),
        #       'AP by rank at Top 5 = ', '{:0.2f}%'.format(average_precision_i_rank_5 * 100))
        #
        # print(metric_names[i], 'AP by score at Top 10 = ', '{:0.2f}%'.format(average_precision_i_score_10 * 100),
        #       'AP by rank  at Top 10 = ', '{:0.2f}%'.format(average_precision_i_rank_10 * 100))
        #
        # print(metric_names[i], prediction_type, ', AP by score at Top 20 = ', '{:0.2f}%'.format(average_precision_i_score_20),
        #       'AP by rank  at Top 20 = ', '{:0.2f}%'.format(average_precision_i_rank_20))
        #
        # print(metric_names[i], prediction_type, ', AP by score at Top 50 = ', '{:0.2f}%'.format(average_precision_i_score_50),
        #       'AP by rank  at Top 50 = ', '{:0.2f}%'.format(average_precision_i_rank_50))
        #
    print('Metric_type\tExact\tW2V_GN:')
    print('AP scores top 5\t', *ap_scores_top5, sep='\t')
    print('AP scores top 10\t', *ap_scores_top10, sep='\t')
    print('AP ranks top 5\t', *ap_rank_top5, sep='\t')
    print('AP ranks top 10\t', *ap_rank_top10, sep='\t')

    Table_score_summary = np.transpose(Table_score_summary)

    Table_rank_summary = np.transpose(Table_rank_summary)

    # table_score_fname = outfile_prefix + '_scores_{}.csv'.format(prediction_type)
    # table_rank_fname = outfile_prefix + '_rank_{}.csv'.format(prediction_type)

    # np.savetxt(table_score_fname, Table_score_summary, delimiter=",", header=','.join(metric_names), comments='')
    # np.savetxt(table_rank_fname, Table_rank_summary, delimiter=",", header=','.join(metric_names), comments='')


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--path', dest='path', required=True,
        help='Detections path')
    parser.add_argument(
        '--model', dest='model', required=True,
        help='Model')


    return parser.parse_args()

root='../'
model_names = os.listdir(root + '/outputs/CV_feat')
model_names = sorted(model_names)
CV_dir = root + '/outputs/CV_feat/'
tembedding_path= root + 'eval/eval_files/TEmbeddings/large_scale/semantic/'
upuntil_flag = True
overall = True

for model_name in model_names:
    if '8tasks_semantic' in model_name:
        print(model_name)
        for b in range(1, 5):
            t_embedding = scipy.io.loadmat(tembedding_path + 'TEmbedding_seen_B' + str(b) + '_test.mat')
            t_embedding = t_embedding['B_T']
            gt = pd.read_csv('./eval_files/gt_words/4tasks_random/' + 'B' + str(b) + '_gt.csv')

            if b == 1:
                t_embedding_all = t_embedding
                gt_all = gt
            else:
                t_embedding_all = np.concatenate([t_embedding_all, t_embedding], axis=0)
                gt_all = gt_all.append(gt)

        for b in range(1, 5):
            if os.path.exists(CV_dir + model_name + '/B' + str(b) + 'XEmbeddings.mat'):
                x_embedding = scipy.io.loadmat(CV_dir + model_name + '/B' + str(b) + 'XEmbeddings.mat')
                x_embedding = x_embedding['XE']

                t_embedding_s = t_embedding_all[:, :300]
                t_embedding_p = t_embedding_all[:, 300:600]
                t_embedding_o = t_embedding_all[:, 600:]

                t_embedding_so = np.concatenate([t_embedding_all[:, :300], t_embedding_all[:, 600:]], axis=0)
                concatvalues = np.concatenate([gt_all.S.values, gt_all.O.values])
                gt_all_SO = pd.DataFrame(concatvalues, columns=['SO'])

                x_embedding_s = x_embedding[:, :300]
                x_embedding_p = x_embedding[:, 300:600]
                x_embedding_o = x_embedding[:, 600:]
                k = 100

                topk_s = evaluate_each(t_embedding_so, x_embedding_s, gt_all_SO['SO'], k)
                topk_p = evaluate_each(t_embedding_p, x_embedding_p, gt_all['P'], k)
                topk_o = evaluate_each(t_embedding_so, x_embedding_o, gt_all_SO['SO'], k)

                print('\nSubjects:')
                do_analysis(10, topk_s)
                print('\nObjects:')
                do_analysis(10, topk_o)
                print('\nPredicates:')
                do_analysis_prd(10, topk_p)
