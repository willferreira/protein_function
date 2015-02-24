import itertools as it
import operator as op
import functools as ft
import math

import numpy as np
import pandas as pd
from nltk.util import ngrams
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from utils import VALID_AMINO_ACIDS


class StatelessTransform(object):
    def fit(self, X, y=None):
        return self


class ExtractSequenceTransform(StatelessTransform):

    def transform(self, X):
        return [str(x.seq) for x in X]


class SequenceLengthTransform(StatelessTransform):

    def transform(self, X):
        arr = np.array([len(s) for s in X]).reshape(len(X), 1)
        return arr


class AminoAcidPercentageTransform(StatelessTransform):

    def transform(self, X):
        vec = np.zeros((len(X), len(VALID_AMINO_ACIDS)))
        for i in range(len(X)):
            pa = ProteinAnalysis(str(X[i]))
            for j, a in enumerate(VALID_AMINO_ACIDS):
                vec[i, j] = pa.get_amino_acids_percent().get(a, 0.0)
        return vec


_MAX_NGRAMS = 2
_ngrams = []
for i in range(2, _MAX_NGRAMS+1):
    _ngrams.extend(it.product(*[VALID_AMINO_ACIDS]*i))
_ngrams = dict([(n, i) for i, n in enumerate(_ngrams)])


class NGramCompositionTransform(StatelessTransform):

    def transform(self, X):
        vec = np.zeros((len(X), len(_ngrams)))
        for i in range(len(X)):
            seq = list(str(X[i]))
            if len(seq) >= _MAX_NGRAMS:
                for n in range(2, _MAX_NGRAMS+1):
                    ngs = ngrams(seq, n)
                    for ng in ngs:
                        vec[i, _ngrams[ng]] += 1
        return vec


class MolecularWeightTransform(StatelessTransform):

    def transform(self, X):
        vec = np.zeros((len(X), 1))
        for i in range(len(X)):
            pa = ProteinAnalysis(str(X[i]))
            vec[i, 0] = pa.molecular_weight()
        return vec


class AromaticityTransform(StatelessTransform):

    def transform(self, X):
        vec = np.zeros((len(X), 1))
        for i in range(len(X)):
            pa = ProteinAnalysis(str(X[i]))
            vec[i, 0] = pa.aromaticity()
        return vec


class InstabilityIndexTransform(StatelessTransform):

    def transform(self, X):
        vec = np.zeros((len(X), 1))
        for i in range(len(X)):
            pa = ProteinAnalysis(str(X[i]))
            vec[i, 0] = pa.instability_index()
        return vec


class FlexibilityTransform(StatelessTransform):

    def transform(self, X):
        vec = np.zeros((len(X), 1))
        for i in range(len(X)):
            pa = ProteinAnalysis(str(X[i]))
            vec[i, 0] = pa.flexibility()
        return vec


class ProteinScaleTransform(StatelessTransform):

    def transform(self, X):
        vec = np.zeros((len(X), 1))
        for i in range(len(X)):
            pa = ProteinAnalysis(str(X[i]))
            vec[i, 0] = pa.protein_scale()
        return vec


class IsoElectricPointTransform(StatelessTransform):

    def transform(self, X):
        vec = np.zeros((len(X), 1))
        for i in range(len(X)):
            pa = ProteinAnalysis(str(X[i]))
            vec[i, 0] = pa.isoelectric_point()
        return vec


class SecondaryStructureFractionTransform(StatelessTransform):

    def transform(self, X):
        vec = np.zeros((len(X), 3))
        for i in range(len(X)):
            pa = ProteinAnalysis(str(X[i]))
            vec[i, :] = np.array(pa.secondary_structure_fraction())
        return vec


class GravyTransform(StatelessTransform):

    def transform(self, X):
        vec = np.zeros((len(X), 1))
        for i in range(len(X)):
            pa = ProteinAnalysis(str(X[i]))
            vec[i, 0] = np.array(pa.gravy())
        return vec


_MAX_WINDOW = 50


class StartEndTransform(StatelessTransform):

    def __init__(self, ratio=0.25):
        self.ratio = ratio
        self.amino_acid_idx = dict([(a, i) for i, a in enumerate(VALID_AMINO_ACIDS)])

    def _set_count(self, seq, m):
        pa = ProteinAnalysis(seq)
        for a, c in pa.count_amino_acids().items():
            m[self.amino_acid_idx[a]] = c

    def transform(self, X):
        m_left = np.zeros((len(X), len(VALID_AMINO_ACIDS)))
        m_right = np.zeros((len(X), len(VALID_AMINO_ACIDS)))

        window = int(len(X)*self.ratio)
        window = min(window, _MAX_WINDOW)

        for i in range(len(X)):
            seq = str(X[i])
            self._set_count(seq[:window], m_left[i, :])
            self._set_count(seq[-window:], m_right[i, :])
        return np.hstack((m_left, m_right))


class ShannonEntropyTransform(StatelessTransform):

    def _calc_shannon_entropy(self, seq):
        pa = ProteinAnalysis(seq)
        return -ft.reduce(op.add,
                          [freq * math.log(freq, 2) for (_, freq) in pa.count_amino_acids().items() if freq > 0], 0.0)

    def transform(self, X):
        vec = np.zeros((len(X), 1))
        for i in range(len(X)):
            vec[i, 0] = self._calc_shannon_entropy(str(X[i]))
        return vec


_AMINO_ACID_PROPERTIES = ["hydrophobic", "polar", "charged", "negative", "small", "tiny", "aromatic", "aliphatic"]

# ACDEFGHIKLMNPQRSTVWY

_taylor_venn_diagram = \
    pd.DataFrame(index=VALID_AMINO_ACIDS, columns=_AMINO_ACID_PROPERTIES,
                 data=[
                        (1, 0, 0, 0, 1, 1, 0, 0),
                        (1, 1, 0, 0, 1, 1, 0, 0),
                        (0, 1, 1, 1, 1, 0, 0, 0),
                        (0, 1, 1, 1, 1, 0, 0, 0),
                        (1, 0, 0, 0, 0, 0, 1, 0),
                        (1, 0, 0, 0, 1, 1, 0, 0),
                        (1, 1, 1, 0, 0, 0, 1, 0),
                        (1, 0, 0, 0, 0, 0, 0, 1),
                        (1, 1, 1, 0, 0, 0, 0, 0),
                        (1, 0, 0, 0, 0, 0, 0, 1),
                        (1, 0, 0, 0, 0, 0, 0, 0),
                        (0, 1, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0, 0, 1, 0, 0, 0),
                        (0, 1, 0, 0, 0, 0, 0, 0),
                        (0, 1, 1, 0, 0, 0, 0, 0),
                        (0, 1, 0, 0, 1, 1, 0, 0),
                        (1, 1, 0, 0, 1, 0, 0, 0),
                        (1, 0, 0, 0, 1, 0, 0, 1),
                        (1, 1, 0, 0, 0, 0, 1, 0),
                        (1, 1, 0, 0, 0, 0, 1, 0)])


class TaylorVennTransform(StatelessTransform):

    def transform(self, X):
        vec = np.zeros((len(X), len(_AMINO_ACID_PROPERTIES)))
        for i in range(len(X)):
            for amino_acid in list(str(X[i])):
                vec[i, :] += _taylor_venn_diagram.ix[amino_acid, :]
        return vec