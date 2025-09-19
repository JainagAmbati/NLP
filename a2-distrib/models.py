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
        embeds = self.embedding(input_batch)                     # (batch_size, seq_len, embed_dim)
        avg_embeds = torch.mean(embeds, dim=1)                   # (batch_size, embed_dim)
        hidden = torch.tanh(self.fc1(self.dropout(avg_embeds)))  # (batch_size, hidden_dim)
        logits = self.fc2(self.dropout(hidden))                  # (batch_size, num_classes)
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
        # Convert words → indices
        indices = []
        for w in ex_words:
            idx = self.word_embeddings.word_indexer.index_of(w)
            if idx == -1:   # unseen word
                idx = 1     # UNK
            indices.append(idx)

        # make into tensor
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            log_probs = self.model(indices_tensor)
        return int(torch.argmax(log_probs, dim=1).item())

class PrefixEmbeddings:
    """
    Prefix-based embeddings: map each word to its prefix (default length=3).
    """
    def __init__(self, base_word_embeddings: WordEmbeddings, prefix_len: int = 3):
        self.prefix_len = prefix_len
        self.word_indexer = Indexer()
        self.word_indexer.add_and_get_index("PAD")  # 0
        self.word_indexer.add_and_get_index("UNK")  # 1

        embed_dim = base_word_embeddings.vectors.shape[1]
        prefix_to_vecs = defaultdict(list)

        # loop over all indices in base embeddings
        for i in range(len(base_word_embeddings.vectors)):
            word = base_word_embeddings.word_indexer.get_object(i)
            if word in ("PAD","UNK"):
                continue
            prefix = word[:prefix_len]
            prefix_to_vecs[prefix].append(base_word_embeddings.vectors[i])

        vectors = [np.zeros(embed_dim), np.zeros(embed_dim)]  # PAD, UNK

        for prefix, vecs in prefix_to_vecs.items():
            avg_vec = np.mean(vecs, axis=0)
            idx = self.word_indexer.add_and_get_index(prefix)
            if idx == len(vectors):
                vectors.append(avg_vec)
            else:
                vectors[idx] = avg_vec

        self.vectors = np.array(vectors, dtype=np.float32)

    def get_initialized_embedding_layer(self, frozen=False) -> nn.Embedding:
        emb = nn.Embedding(len(self.vectors), self.vectors.shape[1], padding_idx=0)
        emb.weight.data.copy_(torch.tensor(self.vectors))
        emb.weight.requires_grad = not frozen
        return emb

    def word_to_prefix_index(self, word: str) -> int:
        prefix = word[:self.prefix_len]
        idx = self.word_indexer.index_of(prefix)
        if idx == -1:   # unseen prefix
            return 1   # UNK
        return idx

class PrefixSentimentClassifier(NeuralSentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        indices = [self.word_embeddings.word_to_prefix_index(w) for w in ex_words]
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            log_probs = self.model(indices_tensor)
        return int(torch.argmax(log_probs, dim=1).item())


def train_deep_averaging_network_old(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize embedding layer
    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=False).to(device)

    # Build model
    hidden_dim = 100
    model = DeepAveragingNetwork(embedding_layer, hidden_dim)
    classifier = NeuralSentimentClassifier(model, word_embeddings, device)

    # Optimizer + loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()

    # Convert examples into (indices, label)
    # def to_tensor(ex: SentimentExample):
    #     indices = [word_embeddings.word_indexer.index_of(w) for w in ex.words]
    #     return torch.tensor(indices, dtype=torch.long), ex.label
    def to_tensor(ex: SentimentExample):
      indices = []
      for w in ex.words:
          idx = word_embeddings.word_indexer.index_of(w)
          if idx == -1:   # word not found
              idx = 1     # UNK
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

            # Pad to max length
            max_len = max(len(seq) for seq in batch_indices)
            padded = [F.pad(seq, (0, max_len - len(seq)), value=0) for seq in batch_indices]
            batch_tensor = torch.stack(padded).to(device)  # (batch, max_len)
            labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            # print(batch_tensor.shape)
            # import sys
            # sys.exit(0)
            log_probs = model(batch_tensor)
            loss = loss_fn(log_probs, labels_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate on dev set
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
 
def train_deep_averaging_network_old2(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Switch embeddings if typo setting is enabled
    if train_model_for_typo_setting:
        word_embeddings = PrefixEmbeddings(word_embeddings)
        classifier_cls = PrefixSentimentClassifier
    else:
        classifier_cls = NeuralSentimentClassifier

    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=False).to(device)
    hidden_dim = 100
    model = DeepAveragingNetwork(embedding_layer, hidden_dim)
    classifier = classifier_cls(model, word_embeddings, device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()

    def to_tensor(ex: SentimentExample):
        if train_model_for_typo_setting:
            indices = [word_embeddings.word_to_prefix_index(w) for w in ex.words]
        else:
            indices = [word_embeddings.word_indexer.index_of(w) for w in ex.words]
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

        # Evaluate
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


import nltk

class SpellCorrectingSentimentClassifier(NeuralSentimentClassifier):
    def __init__(self, model: DeepAveragingNetwork, word_embeddings: WordEmbeddings, device: str = "cpu"):
        super().__init__(model, word_embeddings, device)
        self.vocab = [word_embeddings.word_indexer.get_object(i) 
                      for i in range(len(word_embeddings.word_indexer))]
        self.unk_cache = {}

    def correct_word(self, word: str) -> str:
        # return cached if available
        if word in self.unk_cache:
            return self.unk_cache[word]

        best_word = None
        best_dist = 1e9
        # brute force over vocab (can be optimized if needed)
        for vocab_word in self.vocab:
            if vocab_word in ("PAD", "UNK"):
                continue
            d = nltk.edit_distance(word, vocab_word)
            if d < best_dist:
                best_dist = d
                best_word = vocab_word
        self.unk_cache[word] = best_word if best_word is not None else "UNK"
        return self.unk_cache[word]

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        indices = []
        for w in ex_words:
            idx = self.word_embeddings.word_indexer.index_of(w)
            if idx == -1:  # OOV
                if has_typos:
                    corrected = self.correct_word(w)
                    idx = self.word_embeddings.word_indexer.index_of(corrected)
                if idx == -1:  # still not found
                    idx = 1  # UNK
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
                idx = 1  # UNK
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

        # Dev evaluation
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

