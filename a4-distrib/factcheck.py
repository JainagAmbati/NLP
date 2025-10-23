import torch
from typing import List
import numpy as np
import spacy
import gc
import re
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
STOPWORDS = set(stopwords.words('english'))
LEMM = WordNetLemmatizer()


def _normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = [LEMM.lemmatize(t) for t in text.split() if t not in STOPWORDS]
    return tokens


class FactExample(object):
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label
    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors='pt',
                truncation=True,
                padding=True
            )
            if self.cuda:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            # Model label order: [contradiction, neutral, entailment]
            entail_prob = float(probs[2])

        del inputs, outputs, logits, probs
        gc.collect()
        return entail_prob


class FactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return np.random.choice(["S", "NS"])


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


# class WordRecallThresholdFactChecker(FactChecker):
#     def __init__(self, threshold=0.55):
#         self.threshold = threshold
#         self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

#     def _tokenize(self, text):
#         doc = self.nlp(text.lower())
#         tokens = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
#         return set(tokens)

#     def _recall_overlap(self, fact_tokens, sent_tokens):
#         if not fact_tokens:
#             return 0.0
#         return len(fact_tokens & sent_tokens) / len(fact_tokens)

#     def predict(self, fact: str, passages: List[dict]) -> str:
#         fact_tokens = self._tokenize(fact)
#         if not fact_tokens:
#             return "NS"
#         best_score = 0.0
#         for passage in passages:
#             sents = re.split(r'(?<=[.!?])\s+', passage["text"])
#             for sent in sents:
#                 sent_tokens = self._tokenize(sent)
#                 if not sent_tokens:
#                     continue
#                 score = self._recall_overlap(fact_tokens, sent_tokens)
#                 best_score = max(best_score, score)
#         return "S" if best_score >= self.threshold else "NS"

from nltk.corpus import wordnet as wn

class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def _expand_synonyms(self, lemma):
        syns = {lemma}
        for syn in wn.synsets(lemma):
            for l in syn.lemmas():
                s = l.name().replace("_", " ").lower()
                if s.isalpha() and len(s) > 2:
                    syns.add(s)
        return syns

    def _tokenize(self, text):
        doc = self.nlp(text.lower())
        toks = []
        for t in doc:
            if not t.is_alpha:
                continue
            lemma = t.lemma_
            toks.extend(list(self._expand_synonyms(lemma)))
        return set(toks)

    def _f1_overlap(self, fact_set, sent_set):
        if not fact_set or not sent_set:
            return 0.0
        inter = fact_set & sent_set
        prec = len(inter) / len(sent_set)
        rec = len(inter) / len(fact_set)
        return 2 * prec * rec / (prec + rec + 1e-9)

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = self._tokenize(fact)
        if not fact_tokens:
            return "NS"

        best_score = 0.0
        for passage in passages:
            for sent in re.split(r'(?<=[.!?])\s+', passage["text"]):
                sent_tokens = self._tokenize(sent)
                best_score = max(best_score, self._f1_overlap(fact_tokens, sent_tokens))
        return "S" if best_score >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model, threshold=0.35, overlap_prune=0.08):
        self.ent_model = ent_model
        self.threshold = threshold
        self.overlap_prune = overlap_prune
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def _tokenize(self, text):
        doc = self.nlp(text.lower())
        tokens = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
        return set(tokens)

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = self._tokenize(fact)
        if not fact_tokens:
            return "NS"
        max_prob = 0.0
        for passage in passages:
            sents = re.split(r'(?<=[.!?])\s+', passage["text"])
            for sent in sents:
                sent_tokens = self._tokenize(sent)
                if not sent_tokens:
                    continue
                overlap = len(fact_tokens & sent_tokens) / len(fact_tokens)
                if overlap < self.overlap_prune:
                    continue
                prob = self.ent_model.check_entailment(sent.strip(), fact)
                max_prob = max(max_prob, prob)
        return "S" if max_prob >= self.threshold else "NS"

