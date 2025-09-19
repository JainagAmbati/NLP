# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from sentiment_data import *
from collections import defaultdict


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class DeepAveragingNetwork(nn.Module):
    """
    A simple Deep Averaging Network (DAN).
    Takes averaged embeddings of words in a sentence,
    passes through feedforward layers, outputs log-probs.
    """
    def __init__(self, embedding_layer: nn.Embedding, hidden_dim: int, num_classes: int = 2, dropout: float = 0.3):
        super(DeepAveragingNetwork, self).__init__()
        self.embedding = embedding_layer
        embed_dim = embedding_layer.embedding_dim

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_batch):
        """
        input_batch: (batch_size, seq_len) of word indices
        """
        embeds = self.embedding(input_batch)                     
        avg_embeds = torch.mean(embeds, dim=1)                   
        hidden = torch.relu(self.fc1(self.dropout(avg_embeds)))  
        logits = self.fc2(self.dropout(hidden))                  
        return F.log_softmax(logits, dim=1)

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, model: DeepAveragingNetwork, word_embeddings: WordEmbeddings, device: str = "cpu"):
          self.model = model
          self.word_embeddings = word_embeddings
          self.device = device
          self.model.to(device)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        
        indices = []
        for w in ex_words:
            idx = self.word_embeddings.word_indexer.index_of(w)
            if idx == -1:   
                idx = 1     
            indices.append(idx)

        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            log_probs = self.model(indices_tensor)
        return int(torch.argmax(log_probs, dim=1).item())

import nltk
class SpellCorrectingSentimentClassifier(NeuralSentimentClassifier):
    def __init__(self, model: DeepAveragingNetwork, word_embeddings: WordEmbeddings, device: str = "cpu"):
        super().__init__(model, word_embeddings, device)
        self.vocab = [word_embeddings.word_indexer.get_object(i) 
                      for i in range(len(word_embeddings.word_indexer))]
        self.unk_cache = {}

        self.prefix_dict = {}
        for w in self.vocab:
            if w in ("PAD", "UNK"):
                continue
            prefix = w[:3]
            if prefix not in self.prefix_dict:
                self.prefix_dict[prefix] = []
            self.prefix_dict[prefix].append(w)

    def correct_word(self, word: str) -> str:
        if word in self.unk_cache:
            return self.unk_cache[word]

        prefix = word[:3]
        candidates = self.prefix_dict.get(prefix, [])

        if not candidates:
            L = len(word)
            candidates = [w for w in self.vocab if abs(len(w) - L) <= 2 and w not in ("PAD", "UNK")]

        best_word, best_dist = None, 1e9
        for vocab_word in candidates:
            d = nltk.edit_distance(word, vocab_word)
            if d < best_dist:
                best_dist = d
                best_word = vocab_word

        if best_word is None:
            best_word = "UNK"
        self.unk_cache[word] = best_word
        return best_word

    def preload_corrections(self, all_sentences: List[List[str]]):
        """
        Run once before evaluation to correct all OOV words in dataset.
        This avoids repeated expensive lookups during prediction.
        """
        for sent in all_sentences:
            for w in sent:
                idx = self.word_embeddings.word_indexer.index_of(w)
                if idx == -1: 
                    self.correct_word(w)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        indices = []
        for w in ex_words:
            idx = self.word_embeddings.word_indexer.index_of(w)
            if idx == -1:
                if has_typos:
                    corrected = self.correct_word(w)
                    idx = self.word_embeddings.word_indexer.index_of(corrected)
                if idx == -1:
                    idx = 1
            indices.append(idx)

        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            log_probs = self.model(indices_tensor)
        return int(torch.argmax(log_probs, dim=1).item())


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=False).to(device)
    hidden_dim = 100
    model = DeepAveragingNetwork(embedding_layer, hidden_dim)

    if train_model_for_typo_setting:
        classifier = SpellCorrectingSentimentClassifier(model, word_embeddings, device)
    else:
        classifier = NeuralSentimentClassifier(model, word_embeddings, device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()

    def to_tensor(ex: SentimentExample):
        indices = []
        for w in ex.words:
            idx = word_embeddings.word_indexer.index_of(w)
            if idx == -1:
                idx = 1 
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long), ex.label

    train_data = [to_tensor(ex) for ex in train_exs]
    dev_data = [to_tensor(ex) for ex in dev_exs]

    batch_size = 32
    num_epochs = 10

    for epoch in range(num_epochs):
        random.shuffle(train_data)
        model.train()
        total_loss = 0.0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            batch_indices, batch_labels = zip(*batch)

            max_len = max(len(seq) for seq in batch_indices)
            padded = [F.pad(seq, (0, max_len - len(seq)), value=0) for seq in batch_indices]
            batch_tensor = torch.stack(padded).to(device)
            labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            log_probs = model(batch_tensor)
            loss = loss_fn(log_probs, labels_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ex_tensor, label in dev_data:
                ex_tensor = ex_tensor.unsqueeze(0).to(device)
                pred = torch.argmax(model(ex_tensor), dim=1).item()
                if pred == label:
                    correct += 1
                total += 1
        dev_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss={total_loss:.4f}, Dev Acc={dev_acc:.4f}")

    return classifier

