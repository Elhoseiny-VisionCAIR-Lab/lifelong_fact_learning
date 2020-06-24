import numpy as np

def calcAP(decisionScores, gtlabels, nRetrieval):
    if nRetrieval == 0:
        nRetrieval = len(gtlabels)

    idx = np.argsort(decisionScores)[::-1]
    vals = decisionScores[idx]
    precisions = []
    nTP = 0
    for i in range(nRetrieval):
        if gtlabels[idx[i]] == 1:
            nTP = nTP + 1
            precisions += [nTP / (i + 1)]
    if nTP != 0:
        ap = sum(precisions) / nTP
        pn = precisions[-1]
    else:
        ap = 0
        pn = 0

    return ap, pn
