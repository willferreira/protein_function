from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from transforms import (
    ExtractSequenceTransform,
    AminoAcidPercentageTransform,
    NGramCompositionTransform,
    SequenceLengthTransform,
    MolecularWeightTransform,
    AromaticityTransform,
    InstabilityIndexTransform,
    FlexibilityTransform,
    ProteinScaleTransform,
    IsoElectricPointTransform,
    SecondaryStructureFractionTransform,
    GravyTransform,
    StartEndTransform,
    ShannonEntropyTransform,
    TaylorVennTransform
)

from utils import calc_confusion_matrix


class ProteinFunctionPredictor(object):

    def __init__(self):
        union = make_union(
                           NGramCompositionTransform(),
                           AminoAcidPercentageTransform(),
                           SequenceLengthTransform(),
                           MolecularWeightTransform(),
                           # AromaticityTransform(),
                           InstabilityIndexTransform(),
                           # FlexibilityTransform(),
                           # ProteinScaleTransform(),
                           IsoElectricPointTransform(),
                           SecondaryStructureFractionTransform(),
                           GravyTransform(),
                           StartEndTransform(),
                           ShannonEntropyTransform(),
                           TaylorVennTransform()
                           )
        pipeline = [ExtractSequenceTransform(), union]
        self.pipeline = make_pipeline(*pipeline)
        self.classifier = RandomForestClassifier(**{"n_estimators": 500, "min_samples_leaf": 1, "n_jobs": -1})
        # self.classifier = LogisticRegression(class_weight='auto', dual=False, penalty='l2')
        # self.classifier = OneVsRestClassifier(SVC(kernel='rbf', class_weight='auto', probability=True))

    def fit(self, sequences, y=None):
        """
        Fit the model.
        :param sequences: a list of Bio.SeqRecord.SeqRecord
        :param y: a list of labels from the set {"cyto", "mito", "nucleus", "secreted"}
        :return: self
        """
        Z = self.pipeline.fit_transform(sequences, y)
        self.classifier.fit(Z, y)
        return self

    def predict(self, sequences):
        """
        Return predicted labels for given sequences.
        :param sequences: a list of Bio.SeqRecord.SeqRecord
        :return:
        """
        Z = self.pipeline.transform(sequences)
        labels = self.classifier.predict(Z)
        return labels

    def score(self, sequences, y):
        y_predicted = self.predict(sequences)
        precision = precision_score(y, y_predicted)
        recall = recall_score(y, y_predicted)
        f1 = f1_score(y, y_predicted)
        accuracy = accuracy_score(y, y_predicted)
        cm = calc_confusion_matrix(y, y_predicted)
        return precision, recall, accuracy, f1, cm

