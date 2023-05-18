import numpy as np
from scipy.stats import rankdata

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
		This function computes the average prescision at k between two lists of
		items.
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(y_prob, y_true, k):
    predicted = [np.argsort(p_)[-k:][::-1] for p_ in y_prob]
    actual = [[y_] for y_ in y_true]
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def hits_k(y_prob, y_true, k):
    acc = []
    for p_, y_ in zip(y_prob, y_true):
        top_k = np.argsort(p_)[-k:][::-1]
        acc += [1. if y_ in top_k else 0.]
    return sum(acc) / len(acc)

def mean_rank(y_prob, y):
    ranks = []
    n_classes = y_prob.shape[1]
    for p_, y_ in zip(y_prob, y):
        ranks += [n_classes - rankdata(p_, method='max')[y_]]
    return sum(ranks) / float(len(ranks))

def MRR(y_prob, y_true):
    ranks = []
    n_classes = y_prob.shape[1]
    for p_, y_ in zip(y_prob, y_true):
        ranks += [1 / (n_classes - rankdata(p_, method='max')[y_] + 1)]
    sum_ranks = sum(ranks)
    if type(sum_ranks) == 'list':
        sum_ranks = sum_ranks[0]
    return sum_ranks / float(len(ranks))

def compute_metrics(y_prob, y_true, k_list=[10,50,100]):
    """
    y_prob: (#samples, #users), y_true: (#samples,)
    """
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    scores = {}
    scores['MRR'] = MRR(y_prob, y_true)
    for k in k_list:
        scores['hits@' + str(k)] = hits_k(y_prob, y_true, k=k)
        scores['map@' + str(k)] = mapk(y_prob, y_true, k=k)
    return scores
