
import torch
from typing import List
import numpy as np
import spacy
import gc
import re
from collections import Counter
import math

#we will use spacy-based tokenization and stopwords

try:
    _nlp_for_tokens = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer'])
    _spacy_stopwords = _nlp_for_tokens.Defaults.stop_words
    
except Exception:
    # manually declaring few stopwords if spacy isn't available
    _nlp_for_tokens = None
    _spacy_stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "in", "on", "and", "or", "of", "to",
        "for", "by", "with", "as", "that", "this", "it", "from", "at", "be", "has", "have",
    }
    
_word_tokenizer_re = re.compile(r"[A-Za-z0-9']+")

class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"
    
#define tokenizer and normalizer
def _tokenize_and_normalize(text : str, remove_stopwords= True):
    if text is None:
        return []
    tokens = _word_tokenizer_re.findall(text.lower())
    
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _spacy_stopwords]
    return tokens

#function for jaccard similarity
def _jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0

#calculate tfidf cosine similarity 
def _tfidf_cosine_score(tokens_a, tokens_b):
    terms = sorted(set(tokens_a) | set(tokens_b))
    if not terms:
        return 0.0
    
    idx = {t: i for i, t in enumerate(terms)}
    
    tf_a = np.zeros(len(terms), dtype=float)
    tf_b = np.zeros(len(terms), dtype=float)
    
    #calculate term frequency
    for t in tokens_a:
        tf_a[idx[t]] += 1.0
    for t in tokens_b:
        tf_b[idx[t]] += 1.0
        
    #calculate document frequency
    df = np.zeros(len(terms), dtype=float)
    for i, t in enumerate(terms):
        df[i] = int(t in set(tokens_a)) + int(t in set(tokens_b))
        
    #avoid division by zero
    idf = np.log((2.0 + 1.0) / (df + 1.0)) + 1.0
    
    vec_a = tf_a * idf
    vec_b = tf_b * idf

    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)

class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold: float = 0.65, remove_stopwords: bool = True):
        self.threshold = threshold
        self.remove_stopwords = remove_stopwords
        
    def _score_fact_vs_text(self, fact: str, text: str):
        fact_tokens = _tokenize_and_normalize(fact, remove_stopwords = self.remove_stopwords)
        text_tokens = _tokenize_and_normalize(text, remove_stopwords = self.remove_stopwords)
        set_fact = set(fact_tokens)
        set_text = set(text_tokens)
        
        jacc = _jaccard(set_fact, set_text)
        
        tfidf_cos = _tfidf_cosine_score(fact_tokens, text_tokens)

        # overlap_count / len(fact_tokens_unique)
        recall = 0.0
        if set_fact:
            recall = len(set_fact & set_text) / float(len(set_fact))

        # Combine signals in an empirically simple way: max of the three normalized measures
        combined = max(jacc, tfidf_cos, recall)
        return combined
        
    def predict(self, fact: str, passages: List[dict]) -> str:
        
        if fact is None or not passages: #there is nothing to compare
            return "NS"
        
        best_score = 0.0
        for passage in passages:
            sents = passage.get("text", "")
            score = self._score_fact_vs_text(fact, sents)
            if score > best_score:
                best_score = score
        # cleanup
        gc.collect()
        return "S" if best_score >= self.threshold else "NS"

class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
      with torch.no_grad():
          inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
          if self.cuda:
              inputs = {key: value.to('cuda') for key, value in inputs.items()}
          outputs = self.model(**inputs)
          logits = outputs.logits

      # logits → probabilities
      probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
      entail_prob = probs[0]          # index 0 = entailment
      neutral_prob = probs[1]
      contradiction_prob = probs[2]

      # CLEANUP – prevent memory issues
      del inputs, outputs, logits
      gc.collect()

      return entail_prob, neutral_prob, contradiction_prob

import re
import torch.nn.functional as F

def split_into_sentences_old(text: str):
    # Split on ".", "?", "!" but keep things simple
    sentences = re.split(r'(?<=[.?!])\s+', text)
    # Clean whitespace and drop empty strings
    return [s.strip() for s in sentences if len(s.strip()) > 0]

    #html tags, removing stop words

import re

# Optional: lazy load stopwords when needed
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords', quiet=True)

# import spacy


def split_into_sentences(text: str, remove_stopwords=False, stopword_source="nltk"):
    """
    :param text: string to split
    :param remove_stopwords: bool - whether to remove stopwords
    :param stopword_source: "nltk" or "spacy"
    :return: list of cleaned sentences
    """

    # Load stopwords based on option
    if remove_stopwords:
        if stopword_source.lower() == "nltk":
            stop_set = set(stopwords.words("english"))
        elif stopword_source.lower() == "spacy":
            # lazy-load spacy model only once
            global _spacy_model
            try:
                _spacy_model
            except NameError:
                _spacy_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            stop_set = _spacy_model.Defaults.stop_words
        else:
            raise ValueError("stopword_source must be 'nltk' or 'spacy'")

    # Remove <s> markers
    text = re.sub(r"</?s>", " ", text)

    # Remove brackets but **keep content**
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"[\[\]]", "", text)

    # DO NOT REMOVE NUMBERS (per your request)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Split by . ? !
    sentences = re.split(r'(?<=[.?!])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    # Remove stopwords if chosen
    if remove_stopwords:
        cleaned = []
        for sent in sentences:
            words = sent.split()
            filtered = [w for w in words if w.lower() not in stop_set]
            if filtered:
                cleaned.append(" ".join(filtered))
        return cleaned

    return sentences


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model, entailment_threshold: float = 0.45, prune_word_threshold: float = 0.15):
        self.ent_model = ent_model
        self.entailment_threshold = entailment_threshold
        self.prune_word_threshold = prune_word_threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_words = set(fact.lower().split())
        best_entail_score = 0.0
        print("FACT:",fact)
        for p in passages:

            sentences = split_into_sentences_old(p["text"])

            for sent in sentences:
                # ---- Word overlap pruning ----
                # sent_words = set(sent.lower().split())
                # overlap = len(fact_words & sent_words) / max(len(fact_words), 1)
                # if overlap < self.prune_word_threshold:
                #     continue

                # ---- Run entailment ----
                entail_prob, neutral_prob, contra_prob = self.ent_model.check_entailment(sent, fact)
                
                # print("SENT:",sent,entail_prob, neutral_prob, contra_prob)
                # if entail_prob > best_entail_score:
                #     best_entail_score = entail_prob
                # if best_entail_score >= self.entailment_threshold:
                #   return "S" 
                if entail_prob >= max(neutral_prob, contra_prob):
                  return "S"
                

        # return "S" if best_entail_score >= self.entailment_threshold else "NS"
        return "NS"



# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold: float = 0.65):
        self.nlp = spacy.load('en_core_web_sm')
        self.threshold = threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        if fact is None or not passages:
            return "NS"

        fact_deps = self.get_dependencies(fact)
        if not fact_deps:
            return "NS"

        best_score = 0.0
        for p in passages:
            text = p.get("text", "") or p.get("sent", "") or ""
            doc = self.nlp(text)
            for s in doc.sents:
                sent_deps = self.get_dependencies(s.text)
                if not sent_deps:
                    continue
                inter = len(fact_deps & sent_deps)
                score = inter / float(len(fact_deps))
                if score > best_score:
                    best_score = score
                    if best_score >= 1.0:
                        break
            if best_score >= 1.0:
                break

        # tunable threshold
        #threshold = 0.5
        return "S" if best_score >= threshold else "NS"

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations