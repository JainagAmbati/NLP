# # models.py

# import numpy as np


# class LanguageModel(object):

#     def get_next_char_log_probs(self, context) -> np.ndarray:
#         """
#         Returns a log probability distribution over the next characters given a context.
#         The log should be base e

#         NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
#         layers in TransformerEncoder).
#         :param context: the string context that the LM conditions on
#         :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
#         """
#         raise Exception("Only implemented in subclasses")


#     def get_log_prob_sequence(self, next_chars, context) -> float:
#         """
#         Scores a bunch of characters following context. That is, returns
#         log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
#         The log should be base e

#         NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
#         layers in TransformerEncoder).
#         :param next_chars:
#         :param context:
#         :return: The float probability
#         """
#         raise Exception("Only implemented in subclasses")


# class UniformLanguageModel(LanguageModel):
#     def __init__(self, voc_size):
#         self.voc_size = voc_size

#     def get_next_char_log_probs(self, context):
#         return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

#     def get_log_prob_sequence(self, next_chars, context):
#         return np.log(1.0/self.voc_size) * len(next_chars)


# class NeuralLanguageModel(LanguageModel):
#     def __init__(self):
#         raise Exception("Implement me")

#     def get_next_char_log_probs(self, context):
#         raise Exception("Implement me")

#     def get_log_prob_sequence(self, next_chars, context):
#         raise Exception("Implement me")


# def train_lm(args, train_text, dev_text, vocab_index):
#     """
#     :param args: command-line args, passed through here for your convenience
#     :param train_text: train text as a sequence of characters
#     :param dev_text: dev text as a sequence of characters
#     :param vocab_index: an Indexer of the character vocabulary (27 characters)
#     :return: a NeuralLanguageModel instance trained on the given data
#     """
#     raise Exception("Implement me")

# transformer_lm.py

import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import random

# The Indexer class / utils are part of your assignment framework.
# We only rely on vocab_index.index_of(char) to convert characters to indices.
# If your Indexer has different methods, adjust the code accordingly.

# ---------- PositionalEncoding (same interface as Part 1) ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        super().__init__()
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
      if x.dim() == 3:
          seq_len = x.size(-2) if self.batched else x.size(0)
      else:
          seq_len = x.size(0)
      seq_len = min(seq_len, self.emb.num_embeddings)  # prevent overflow
      device = x.device
      indices = torch.arange(seq_len, device=device)
      pos_emb = self.emb(indices)
      if x.dim() == 3:
          if self.batched:
              return x + pos_emb.unsqueeze(0)
          else:
              return x + pos_emb.unsqueeze(1)
      else:
          return x + pos_emb



# ---------- Helper to get vocab size safely ----------
def get_vocab_size_from_indexer(vocab_index):
    # Try common attributes
    if hasattr(vocab_index, 'size'):
        try:
            return vocab_index.size()
        except:
            pass
    if hasattr(vocab_index, 'get_size'):
        try:
            return vocab_index.get_size()
        except:
            pass
    if hasattr(vocab_index, 'num_words'):
        return vocab_index.num_words
    # fallback: try internal list (may work in the assignment utils)
    if hasattr(vocab_index, '_index_to_word'):
        return len(vocab_index._index_to_word)
    # Last resort: assume 27 (as assignment says)
    return 27


# ---------- PyTorch Transformer LM Module ----------
class TorchTransformerLM(nn.Module):
    def __init__(self, vocab_size, seq_len=20, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.device = device

        self.embedding = nn.Embedding(vocab_size, d_model)
        # Allow for sequences up to 256 positions safely
        self.pos_enc = PositionalEncoding(d_model, num_positions=256, batched=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        # small initialization for faster convergence
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0.0)

    def generate_causal_mask(self, seq_len):
        # PyTorch Transformer expects mask of shape [seq_len, seq_len] with float values where
        # positions that are masked are float('-inf'). We'll use a triangular mask.
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        return mask.to(self.embedding.weight.device)

    def forward(self, src, src_mask=None):
        """
        src: [seq_len, batch_size] LongTensor of token indices
        src_mask: [seq_len, seq_len] optional mask (float where masked positions are -inf)
        returns: logits [seq_len, batch_size, vocab_size]
        """
        # Embed: -> [seq_len, batch, d_model]
        emb = self.embedding(src) * math.sqrt(self.d_model)  # [seq_len, batch, d_model]
        emb = self.pos_enc(emb)  # add positional enc
        # TransformerEncoder expects [seq_len, batch, d_model] when batch_first=False
        # Apply encoder
        enc = self.encoder(emb, mask=src_mask)  # [seq_len, batch, d_model]
        logits = self.output_proj(enc)  # [seq_len, batch, vocab_size]
        return logits


# ---------- LanguageModel classes ----------
class LanguageModel(object):
    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, torch_model: TorchTransformerLM, vocab_index, seq_len=20, device='cpu'):
        """
        Wraps a trained TorchTransformerLM and the vocabulary indexer.
        """
        self.torch_model = torch_model
        self.vocab_index = vocab_index
        self.seq_len = seq_len
        self.device = device
        # Put model in eval mode for deterministic probability outputs
        self.torch_model.to(self.device)
        self.torch_model.eval()

    def _str_to_indices(self, s: str):
        # convert string to list of indices using vocab_index.index_of
        return [self.vocab_index.index_of(c) for c in s]

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Given a context string, return a numpy vector of log probabilities for the next character.
        Strategy (matches assignment suggestion):
          - Use a fixed chunk size self.seq_len.
          - Build the input to the Transformer as: [space] + last (seq_len-1) characters of context.
          - Feed into the model and take the predicted distribution at the final position.
        """
        self.torch_model.eval()

        # Ensure context is a python string
        if context is None:
            context = ""

        # Prepare input_chunk: length seq_len
        # Take last (seq_len - 1) characters of context (or pad left with spaces)
        tail = context[-(self.seq_len - 1):] if len(context) >= (self.seq_len - 1) else context
        pad_len = (self.seq_len - 1) - len(tail)
        # use space as start-of-sequence / padding as in assignment
        space_char = ' '
        input_str = (space_char * pad_len) + tail  # length seq_len-1
        input_str = space_char + input_str  # prepend a space to make length seq_len

        indices = torch.LongTensor(self._str_to_indices(input_str)).to(self.device)  # [seq_len]
        src = indices.unsqueeze(1)  # [seq_len, 1]
        # create causal mask
        src_mask = self.torch_model.generate_causal_mask(self.seq_len).to(self.device)

        with torch.no_grad():
            logits = self.torch_model(src, src_mask=src_mask)  # [seq_len, 1, vocab_size]
            final_logits = logits[-1, 0, :]  # logits for the last position
            log_probs = torch.log_softmax(final_logits, dim=-1)  # stable log-probs
            return log_probs.cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Computes log P(next_chars | context) = sum_t log P(next_char_t | context + previous next_chars)
        We'll iteratively call get_next_char_log_probs and sum logs.
        """
        total_logp = 0.0
        cur_context = context
        for ch in next_chars:
            logprobs = self.get_next_char_log_probs(cur_context)
            idx = self.vocab_index.index_of(ch)
            total_logp += float(logprobs[idx])
            cur_context = cur_context + ch
        return total_logp


# ---------- Training function ----------
def train_lm(args, train_text: str, dev_text: str, vocab_index):
    """
    Trains a Transformer language model on the given train_text and returns a NeuralLanguageModel wrapper.
    Assumptions:
      - We use chunk_size = 20 (seq_len)
      - We follow the assignment instruction: when given a chunk of 20 chars, feed ' ' + first 19 and predict 20 chars.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seq_len = 20  # as in assignment
    vocab_size = get_vocab_size_from_indexer(vocab_index)
    print(f"[train_lm] device={device} vocab_size={vocab_size}")

    # Hyperparameters (tuned to be reasonable for the assignment)
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 256
    dropout = 0.1
    batch_size = 64
    lr = 1e-3
    num_epochs = 6  # usually enough per assignment hints; adjust if needed
    print_interval = 1000

    # Build model
    torch_model = TorchTransformerLM(vocab_size=vocab_size,
                                     seq_len=seq_len,
                                     d_model=d_model,
                                     nhead=nhead,
                                     num_layers=num_layers,
                                     dim_feedforward=dim_feedforward,
                                     dropout=dropout,
                                     device=device).to(device)

    # Prepare data as overlapping chunks:
    # For each position i in [0, len(train_text)-seq_len], take chunk = train_text[i:i+seq_len]
    # Input to model will be: ' ' + chunk[:seq_len-1]  (length seq_len)
    # Targets will be chunk (length seq_len)
    train_chunks = []
    for i in range(0, len(train_text) - seq_len):
        chunk = train_text[i:i + seq_len]
        # Skip chunks that contain characters not in vocab (unlikely)
        train_chunks.append(chunk)

    # For dev evaluation, form dev_chunks similarly
    dev_chunks = []
    for i in range(0, len(dev_text) - seq_len):
        chunk = dev_text[i:i + seq_len]
        dev_chunks.append(chunk)

    print(f"[train_lm] Train chunks: {len(train_chunks)}, Dev chunks: {len(dev_chunks)}")

    # Convert chunks to numeric tensors (we will create two tensors: inputs and targets)
    def chunk_to_input_target(chunk):
        # chunk: length seq_len
        # input_str = ' ' + chunk[:seq_len-1]
        input_str = ' ' + chunk[:seq_len - 1]
        target_str = chunk  # predict the characters in chunk
        input_ids = [vocab_index.index_of(c) for c in input_str]
        target_ids = [vocab_index.index_of(c) for c in target_str]
        return input_ids, target_ids

    train_inputs = []
    train_targets = []
    for ch in train_chunks:
        inp, tgt = chunk_to_input_target(ch)
        train_inputs.append(inp)
        train_targets.append(tgt)
    train_inputs = np.array(train_inputs, dtype=np.int64)  # [N, seq_len]
    train_targets = np.array(train_targets, dtype=np.int64)  # [N, seq_len]

    # Create batched datasets (drop final partial batch)
    num_train = train_inputs.shape[0]
    num_batches = num_train // batch_size
    if num_batches == 0:
        batch_size = max(1, num_train)
        num_batches = 1

    # optimizer and loss
    optimizer = optim.Adam(torch_model.parameters(), lr=lr)
    loss_fcn = nn.CrossEntropyLoss()  # expects logits (not log_probs)

    # Precompute causal mask
    causal_mask = torch_model.generate_causal_mask(seq_len).to(device)

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        perm = np.random.permutation(num_train)
        torch_model.train()
        start = time.time()
        for b in range(num_batches):
            batch_idx = perm[b * batch_size:(b + 1) * batch_size]
            batch_inputs = torch.LongTensor(train_inputs[batch_idx]).to(device)  # [B, seq_len]
            batch_targets = torch.LongTensor(train_targets[batch_idx]).to(device)  # [B, seq_len]

            # Transformer expects [seq_len, batch]
            src = batch_inputs.transpose(0, 1).contiguous()  # [seq_len, B]
            logits = torch_model(src, src_mask=causal_mask)  # [seq_len, B, vocab_size]

            # reshape for loss: [seq_len * B, vocab_size] and targets [seq_len * B]
            logits_flat = logits.view(seq_len * batch_size, vocab_size)
            targets_flat = batch_targets.transpose(0, 1).contiguous().view(seq_len * batch_size)

            optimizer.zero_grad()
            loss = loss_fcn(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()

            step_loss = float(loss.item())
            epoch_loss += step_loss
            global_step += 1

            if global_step % print_interval == 0:
                print(f"[train_lm] epoch {epoch+1} step {global_step} loss {step_loss:.4f}")

        elapsed = time.time() - start
        print(f"[train_lm] Epoch {epoch+1}/{num_epochs} finished. avg loss {epoch_loss / max(1, num_batches):.4f}. time {elapsed:.2f}s")

        # Evaluate perplexity on dev set (we'll compute average negative log-likelihood per token)
        torch_model.eval()
        total_logp = 0.0
        total_tokens = 0
        with torch.no_grad():
            # process dev in batches of batch_size (pad if needed)
            dev_num = len(dev_inputs := [])
            # create dev arrays if any dev_chunks exist
            if len(dev_chunks) > 0:
                dev_inputs = []
                dev_targets = []
                for ch in dev_chunks:
                    inp, tgt = chunk_to_input_target(ch)
                    dev_inputs.append(inp)
                    dev_targets.append(tgt)
                dev_inputs = np.array(dev_inputs, dtype=np.int64)
                dev_targets = np.array(dev_targets, dtype=np.int64)
                D = dev_inputs.shape[0]
                d_batch_size = batch_size
                d_num_batches = max(1, D // d_batch_size)
                for db in range(d_num_batches):
                    bidx = range(db * d_batch_size, (db + 1) * d_batch_size)
                    batch_inputs = torch.LongTensor(dev_inputs[list(bidx)]).to(device)  # [B, seq_len]
                    batch_targets = torch.LongTensor(dev_targets[list(bidx)]).to(device)
                    src = batch_inputs.transpose(0, 1).contiguous()
                    logits = torch_model(src, src_mask=causal_mask)  # [seq_len, B, V]
                    logp = torch.log_softmax(logits, dim=-1)  # [seq_len, B, V]
                    # gather probabilities for targets
                    # targets shape [seq_len, B]
                    t = batch_targets.transpose(0, 1).contiguous()  # [seq_len, B]
                    # sum log probs
                    logp_for_targets = logp.gather(dim=2, index=t.unsqueeze(2)).squeeze(2)  # [seq_len, B]
                    total_logp += float(logp_for_targets.sum().item())
                    total_tokens += (seq_len * d_batch_size)

        if total_tokens > 0:
            avg_neg_ll = - total_logp / total_tokens
            ppl = math.exp(avg_neg_ll)
            print(f"[train_lm] Dev avg neg log-lik per token = {avg_neg_ll:.4f}, perplexity = {ppl:.4f}")
        else:
            print("[train_lm] No dev tokens to evaluate.")

    # After training, wrap into NeuralLanguageModel and return
    model_wrapper = NeuralLanguageModel(torch_model, vocab_index, seq_len=seq_len, device=device)
    return model_wrapper
