# models.py

import numpy as np
import random
from collections import Counter

from sentiment_data import *
from utils import *


class FeatureExtractor(object):
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        for word in sentence:
            word = word.lower()
            idx = self.indexer.add_and_get_index("UNI=%s" % word, add=add_to_indexer)
            if idx != -1:
                feats[idx] += 1
        return feats


class BigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        lowered = [w.lower() for w in sentence]
        for i in range(len(lowered) - 1):
            bigram = lowered[i] + "|" + lowered[i + 1]
            idx = self.indexer.add_and_get_index("BIGRAM=%s" % bigram, add=add_to_indexer)
            if idx != -1:
                feats[idx] += 1
        return feats


class BetterFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        lowered = [w.lower() for w in sentence if w.isalpha()]
        for word in lowered:
            if len(word) > 2:  # filter short words
                idx = self.indexer.add_and_get_index("UNI=%s" % word, add=add_to_indexer)
                if idx != -1:
                    feats[idx] += 1
        for i in range(len(lowered) - 1):
            bigram = lowered[i] + "|" + lowered[i + 1]
            idx = self.indexer.add_and_get_index("BIGRAM=%s" % bigram, add=add_to_indexer)
            if idx != -1:
                feats[idx] += 1
        return feats


class SentimentClassifier(object):
    def predict(self, sentence: List[str]) -> int:
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[idx] * val for idx, val in feats.items())
        return 1 if score >= 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[idx] * val for idx, val in feats.items())
        prob = 1.0 / (1.0 + np.exp(-score))
        return 1 if prob >= 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs=5, lr=1.0) -> PerceptronClassifier:
    indexer = feat_extractor.get_indexer()
    # First pass: build feature vocab
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    weights = np.zeros(len(indexer))

    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[idx] * val for idx, val in feats.items())
            pred = 1 if score >= 0 else 0
            if pred != ex.label:
                update = lr * (ex.label - pred)
                for idx, val in feats.items():
                    weights[idx] += update * val
    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs=10, lr=0.1) -> LogisticRegressionClassifier:
    indexer = feat_extractor.get_indexer()
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    weights = np.zeros(len(indexer))

    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[idx] * val for idx, val in feats.items())
            prob = 1.0 / (1.0 + np.exp(-score))
            error = ex.label - prob
            for idx, val in feats.items():
                weights[idx] += lr * error * val
    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
