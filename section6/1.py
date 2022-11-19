import numpy as np


def tfidf(n, N, D, d):
    return n/N * np.log(D/d)

# [['1 1 2']['1 2']['3']]

res = np.zeros((3, 3))

# 1st
res[0, 0] = tfidf(2, 3, 3, 2)
res[0, 1] = tfidf(1, 3, 3, 2)
res[0, 2] = tfidf(0, 3, 3, 1)

# 2nd
res[1, 0] = tfidf(1, 2, 3, 2)
res[1, 1] = tfidf(1, 2, 3, 2)
res[1, 2] = tfidf(0, 2, 3, 1)

# 2nd
res[2, 0] = tfidf(0, 1, 3, 2)
res[2, 1] = tfidf(0, 1, 3, 2)
res[2, 2] = tfidf(1, 1, 3, 1)

print(res.sum())