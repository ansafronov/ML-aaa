import numpy as np


class TfidfVectorizer:
    def __init__(self):
        self.sorted_vocab = {}

    def fit(self, X):
        X_split = []
        for x in X:
            X_split.append(x.split())
        X_split = np.array(X_split)

        self.sorted_vocab = np.sort(X_split.ravel().unique())

        return self

    def transform(self, X):
        pass


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