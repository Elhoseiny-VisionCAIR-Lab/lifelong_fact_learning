import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import json

import nltk
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

import os
import sys
import csv

import gensim
from numpy import linalg as la

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')


def load_w2v_model(path):
    # word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./eval_files/word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    print('Model loaded.')
    # change everything into lowercase
    all_keys = list(word2vec_model.vocab.keys())
    for key in all_keys:
        new_key = key.lower()
        word2vec_model.vocab[new_key] = word2vec_model.vocab.pop(key)
    temp_dict = {x.replace('_', '-'): y for x, y in word2vec_model.vocab.items()}
    word2vec_model.vocab.update(temp_dict)
    print('GoogleNews words converted to lowercase.')
    return word2vec_model


def get_w2v_vecs(categories, word2vec_model):
    # represent background with the word 'unknown'
    # obj_cats.insert(0, 'unknown')
    vecs_dict = {obj: np.zeros(300, dtype=np.float32) for obj in categories}
    vecs = np.zeros((len(categories), 300), dtype=np.float32)
    for r, obj_cat in enumerate(categories):
        obj_words = obj_cat.split('_')
        for word in obj_words:
            if word == 'frenchhorn':
                raw_vec = word2vec_model['french_horn']
            elif word == 'TV':
                raw_vec = word2vec_model['television']
            else:
                raw_vec = word2vec_model[word]
            vecs[r] += (raw_vec / la.norm(raw_vec))
            vecs_dict[obj_cat] += (raw_vec / la.norm(raw_vec))
        vecs[r] /= len(obj_words)
        vecs_dict[obj_cat] /= len(obj_words)
    print('Object label vectors loaded.')
    return vecs_dict


def wordNetsense_Score(sense1, sense2, i):
    if (sense1.pos() in ['s', 'a'] or sense2.pos() in ['s', 'a']):
        return 0
    if (sense1.pos() != sense2.pos()):
        return 0
    if (i == 'lch'):
        return sense1.lch_similarity(sense2)
    elif (i == 'wup'):
        return sense1.wup_similarity(sense2)
    elif (i == 'res'):
        return sense1.res_similarity(sense2, brown_ic)
    elif (i == 'jcn'):
        return sense1.jcn_similarity(sense2, brown_ic)
    elif (i == 'lin'):
        return sense1.lin_similarity(sense2, brown_ic)
    elif (i == 'path'):
        return sense1.path_similarity(sense2)


def wordNetWord_Score(word1, word2, i):
    score = -10000
    ret_s1 = None
    ret_s2 = None
    for s1 in wn.synsets(word1):
        for s2 in wn.synsets(word2):
            s1_s2_sim = wordNetsense_Score(s1, s2, i)
            if not s1_s2_sim is None:
                if (s1_s2_sim > score):
                    score = s1_s2_sim
                    ret_s1 = s1
                    ret_s2 = s2
    return score, ret_s1, ret_s2


def get_sim_matrix(obj_categories, sim_type):
    obj_data = {key: [0.0 for _ in range(len(obj_categories))] for key in obj_categories}
    obj_sim_df = pd.DataFrame(obj_data, index=obj_categories, columns=obj_categories)

    for i in tqdm(range(len(obj_categories))):
        obj1 = obj_categories[i]
        for obj2 in obj_categories:
            obj_sim_df.loc[obj1, obj2], _, _ = wordNetWord_Score(obj1, obj2, sim_type)
    return obj_sim_df


def get_w2v_sim_matrix(categories, vecs_dict):
    data = {key: [0.0 for _ in range(len(categories))] for key in categories}
    sim_df = pd.DataFrame(data, index=categories, columns=categories)

    for i in tqdm(range(len(categories))):
        obj1 = categories[i]
        for obj2 in categories:
            sim_df.loc[obj1, obj2] = np.dot(vecs_dict[obj1], vecs_dict[obj2].transpose())
    return sim_df


obj_categories = pd.read_csv('4tasks_random/SO.csv', header=None).values.tolist()
obj_categories = [elem[0] for elem in obj_categories]
# print('obj_categories', (obj_categories))

prd_categories = pd.read_csv('4tasks_random/P.csv', header=None).values.tolist()
prd_categories = [elem[0] for elem in prd_categories]
# print('prd_categories', (prd_categories))

#print(prd_categories)

# obj_synsets_csv = pd.read_csv('objects_synsets.csv')
# print(obj_synsets['object_name'])
# obj_synsets = dict(zip(obj_synsets_csv['object_name'], obj_synsets_csv['Sherif_synset']))
# print(obj_synsets)

# prd_synsets = json.load(open('./words_synsets_prd.json'))['verbs']

# zeros_prd = [0.0 for _ in range(len(prd_categories))]
# prd_data = {key: zeros_prd for key in prd_categories}

# prd_sim_df = pd.DataFrame(prd_data, index=prd_categories, columns=prd_categories)

# obj_synsets_obj = {obj: wn.synset(obj_synsets[obj]) for obj in obj_categories}
# prd_synsets_obj = {prd: wn.synset(prd_synsets[prd]) for prd in prd_categories}
#print(obj_synsets_obj)

# obj_sim_df = get_sim_matrix(obj_categories, obj_synsets_obj, 'lch')
# #print(obj_sim_df)
# print('Saving')
# obj_sim_df.to_pickle('./similarity_matrices/obj_sim_lch.pkl')
# print('Done')
#
# obj_sim_df = get_sim_matrix(obj_categories, obj_synsets_obj, 'wup')
# #print(obj_sim_df)
# print('Saving')
# obj_sim_df.to_pickle('./similarity_matrices/obj_sim_wup.pkl')
# print('Done')
#
# obj_sim_df = get_sim_matrix(obj_categories, obj_synsets_obj, 'res')
# #print(obj_sim_df)
# print('Saving')
# obj_sim_df.to_pickle('./similarity_matrices/obj_sim_res.pkl')
# print('Done')
#
# obj_sim_df = get_sim_matrix(obj_categories, obj_synsets_obj, 'jcn')
# #print(obj_sim_df)
# print('Saving')
# obj_sim_df.to_pickle('./similarity_matrices/obj_sim_jcn.pkl')
# print('Done')
#
# obj_sim_df = get_sim_matrix(obj_categories, obj_synsets_obj, 'lin')
# #print(obj_sim_df)
# print('Saving')
# obj_sim_df.to_pickle('./similarity_matrices/obj_sim_lin.pkl')
# print('Done')
#
# obj_sim_df = get_sim_matrix(obj_categories, obj_synsets_obj, 'path')
# #print(obj_sim_df)
# print('Saving')
# obj_sim_df.to_pickle('./similarity_matrices/obj_sim_path.pkl')
# print('Done')

obj_sim_df = get_sim_matrix(obj_categories, 'lch')
print('Saving')
obj_sim_df.to_pickle('./similarity_matrices/obj_sim_lch.pkl')
print('Done')

obj_sim_df = get_sim_matrix(obj_categories, 'wup')
print('Saving')
obj_sim_df.to_pickle('./similarity_matrices/obj_sim_wup.pkl')
print('Done')

obj_sim_df = get_sim_matrix(obj_categories, 'res')
print('Saving')
obj_sim_df.to_pickle('./similarity_matrices/obj_sim_res.pkl')
print('Done')

obj_sim_df = get_sim_matrix(obj_categories, 'jcn')
print('Saving')
obj_sim_df.to_pickle('./similarity_matrices/obj_sim_jcn.pkl')
print('Done')

obj_sim_df = get_sim_matrix(obj_categories, 'lin')
print('Saving')
obj_sim_df.to_pickle('./similarity_matrices/obj_sim_lin.pkl')
print('Done')

obj_sim_df = get_sim_matrix(obj_categories, 'path')
print('Saving')
obj_sim_df.to_pickle('./similarity_matrices/obj_sim_path.pkl')
print('Done')

word2vec_model_gn = load_w2v_model('./eval_files/word2vec_model/GoogleNews-vectors-negative300.bin')
word2vec_model_relco = load_w2v_model('./eval_files/word2vec_model/vg_300d_skipgram_rel')

obj_vecs_dict = get_w2v_vecs(obj_categories, word2vec_model_gn)
obj_w2v_sim_df = get_w2v_sim_matrix(obj_categories, obj_vecs_dict)
print('Saving')
obj_w2v_sim_df.to_pickle('./similarity_matrices/obj_sim_w2v_gn.pkl')
print('Done')

obj_vecs_dict = get_w2v_vecs(obj_categories, word2vec_model_relco)
obj_w2v_sim_df = get_w2v_sim_matrix(obj_categories, obj_vecs_dict)
print('Saving')
obj_w2v_sim_df.to_pickle('./similarity_matrices/obj_sim_w2v_relco.pkl')
print('Done')

prd_vecs_dict = get_w2v_vecs(prd_categories, word2vec_model_gn)
prd_w2v_sim_df = get_w2v_sim_matrix(prd_categories, prd_vecs_dict)
print('Saving')
prd_w2v_sim_df.to_pickle('./similarity_matrices/prd_sim_w2v_gn.pkl')
print('Done')

prd_vecs_dict = get_w2v_vecs(prd_categories, word2vec_model_relco)
prd_w2v_sim_df = get_w2v_sim_matrix(prd_categories, prd_vecs_dict)
print('Saving')
prd_w2v_sim_df.to_pickle('./similarity_matrices/prd_sim_w2v_relco.pkl')
print('Done')
