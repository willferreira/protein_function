import os
import functools as ft

import numpy as np
import pandas as pd

from Bio import SeqIO

LABELS = ["cyto", "mito", "nucleus", "secreted"]
VALID_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
INVALID_AMINO_ACIDS = set("BJOUXZ")


@ft.lru_cache(maxsize=None)
def load_training_data():
    print("Loading protein training data...")
    y = []
    X = []
    for label in LABELS:
        sequences = SeqIO.parse(os.path.join("data", "{0:s}.fasta".format(label)), "fasta")
        sequences = list(filter(lambda s: INVALID_AMINO_ACIDS.isdisjoint(set(s.seq)), sequences))
        y.extend([label]*len(sequences))
        X.extend(sequences)
    print("done.")
    return np.array(X), np.array(y)


@ft.lru_cache(maxsize=None)
def load_test_data():
    print("Loading protein test data...")
    data = np.array(SeqIO.parse(os.path.join("data", "blind.fasta"), "fasta"))
    print("done.")
    return data


def calc_confusion_matrix(actual, predicted):
    cm = pd.DataFrame(index=LABELS, columns=LABELS, data=0.0)
    for i in range(len(actual)):
        cm.ix[actual[i], predicted[i]] += 1
    return cm
