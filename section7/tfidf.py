import numpy as np


class TfidfVectorizer:
    def __init__(self):
        self.sorted_vocab = set()
        self.idf = dict()

    def fit(self, X):
        X_elements = set()
        D = len(X)
        d = {}
        for x in X:
            x_split = x.split()
            for i, el in enumerate(x_split):
                X_elements.add(el)
                if el not in x_split[:i]:
                    if el in d.keys():
                        d[el] += 1
                    else:
                        d[el] = 1

        self.sorted_vocab = sorted(X_elements)

        for voc in self.sorted_vocab:
            self.idf[voc] = np.log(D / d[voc])

        return self

    def transform(self, X) -> np.array:
        X_len = len(X)

        N = np.zeros((X_len, len(self.sorted_vocab)))
        n = np.zeros((X_len, len(self.sorted_vocab)))
        idf = np.repeat([[self.idf[voc] for voc in self.sorted_vocab]],
                        X_len, axis=0)

        X_transformed = []
        for i, x in enumerate(X):
            x_split = x.split()
            N[i, :] = len(x_split)
            for j, voc in enumerate(self.sorted_vocab):
                if voc in x_split:
                    n[i, j] = x_split.count(voc)

        X_transformed = n / N * idf

        return X_transformed


def read_input():
    n1, n2 = map(int, input().split())

    train_texts = [input().strip() for _ in range(n1)]
    test_texts = [input().strip() for _ in range(n2)]

    return train_texts, test_texts


def solution():
    train_texts, test_texts = read_input()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_texts)
    transformed = vectorizer.transform(test_texts)

    for row in transformed:
        row_str = ' '.join(map(str, np.round(row, 3)))
        print(row_str)


solution()
