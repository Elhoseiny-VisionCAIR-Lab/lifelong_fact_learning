# Generated with SMOP  0.41
# from libsmop import *
# get_exact_cos.m
import numpy as np
from sklearn.neighbors import NearestNeighbors

def get_exact_cos(x=None,x_test=None,K=None,*args,**kwargs):
    K = min(K, x.shape[0])
    # varargin = get_exact_cos.varargin
    # nargin = get_exact_cos.nargin

    # norm_X_embedding=np.sum(x ** 2, axis=1, keepdims=True) ** (0.5)
    # norm_X_embedding = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
# get_exact_cos.m:2
#     X_embeddingNorm=bsxfun(rdivide,x,norm_X_embedding)
    X_embeddingNorm = x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)
# get_exact_cos.m:3
#     norm_x=sum(x_test ** 2,2) ** (0.5)
#     norm_x=np.linalg.norm(x_test, ord=2, axis=1, keepdims=True)
# get_exact_cos.m:4
#     x_Norm=bsxfun(rdivide,x_test,norm_x)
    x_Norm=x_test/np.linalg.norm(x_test, ord=2, axis=1, keepdims=True)
# get_exact_cos.m:5
#     ind,D=knnsearch(X_embeddingNorm,x_Norm,'K',K,nargout=2)
    # get_exact_cos.m:6

    nbrs = NearestNeighbors(n_neighbors=K, algorithm='brute').fit(X_embeddingNorm)
    D, ind = nbrs.kneighbors(x_Norm)
    return ind, D
