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
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        return 1


class DeepAveragingNetwork(nn.Module):
    def __init__(self, embedding_layer: nn.Embedding, hidden_dim: int, num_classes: int = 2, dropout: float = 0.3):
        super(DeepAveragingNetwork, self).__init__()
        self.embedding = embedding_layer
        embed_dim = embedding_layer.embedding_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_batch):
        embeds = self.embedding(input_batch)                     # (batch, seq_len, embed_dim)
        avg_embeds = torch.mean(embeds, dim=1)                   # (batch, embed_dim)
        hidden = torch.tanh(self.fc1(self.dropout(avg_embeds)))  # (batch, hidden_dim)
        logits = self.fc2(self.dropout(hidden))                  # (batch, num_classes)
        return F.log_softmax(logits, dim=1)


class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, model: DeepAveragingNetwork, word_embeddings, device: str = "cpu"):
        self.model = model
        self.word_embeddings = word_embeddings
        self.device = device
        self.model.to(device)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        indices = [self.word_embeddings.word_indexer.index_of(w) for w in ex_words]
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            log_probs = self.model(indices_tensor)
        return int(torch.argmax(log_probs, dim=1).item())


# ===========================
# Q3: Prefix Embeddings
# ===========================

class PrefixEmbeddings:
    """
    Similar to WordEmbeddings but uses 3-char prefixes.
    """
    def __init__(self, base_word_embeddings: WordEmbeddings, prefix_len: int = 3):
        self.prefix_len = prefix_len
        self.word_indexer = Indexer()
        self.word_indexer.add_and_get_index(PAD_TOKEN)  # 0
        self.word_indexer.add_and_get_index(UNK_TOKEN)  # 1

        # collect prefixes
        prefix_to_vecs = defaultdict(list)
        for i, word in enumerate(base_word_embeddings.word_indexer.objs):
            if i < 2:  # skip PAD/UNK
                continue
            prefix = word[:prefix_len]
            prefix_to_vecs[prefix].append(base_word_embeddings.vectors[i])

        self.vectors = []
        self.vectors.append(np.zeros(base_word_embeddings.vectors.shape[1]))  # PAD
        self.vectors.append(np.zeros(base_word_embeddings.vectors.shape[1]))  # UNK

        for prefix, vecs in prefix_to_vecs.items():
            avg_vec = np.mean(vecs, axis=0)
            self.word_indexer.add_and_get_index(prefix)
            self.vectors.append(avg_vec)

        self.vectors = np.array(self.vectors)

    def get_initialized_embedding_layer(self, frozen=False) -> nn.Embedding:
        emb = nn.Embedding(self.vectors.shape[0], self.vectors.shape[1], padding_idx=0)
        emb.weight.data.copy_(torch.tensor(self.vectors, dtype=torch.float))
        emb.weight.requires_grad = not frozen
        return emb

    def word_to_prefix_index(self, word: str) -> int:
        prefix = word[:self.prefix_len]
        idx = self.word_indexer.index_of(prefix)
        if idx == -1:
            return 1  # UNK
        return idx


class PrefixSentimentClassifier(NeuralSentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        indices = [self.word_embeddings.word_to_prefix_index(w) for w in ex_words]
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            log_probs = self.model(indices_tensor)
        return int(torch.argmax(log_probs, dim=1).item())


# ===========================
# Training
# ===========================

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
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
