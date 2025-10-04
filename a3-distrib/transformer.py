# # transformer.py

# import time
# import torch
# import torch.nn as nn
# import numpy as np
# import random
# from torch import optim
# import matplotlib.pyplot as plt
# from typing import List
# from utils import *


# # Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# # a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# # of it (output_tensor).
# # Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# # times previously in the input sequence (not counting the current occurrence).
# class LetterCountingExample(object):
#     def __init__(self, input: str, output: np.array, vocab_index: Indexer):
#         self.input = input
#         self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
#         self.input_tensor = torch.LongTensor(self.input_indexed)
#         self.output = output
#         self.output_tensor = torch.LongTensor(self.output)


# # Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# # a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# # to return distributions over the labels (0, 1, or 2).
# class Transformer(nn.Module):
#     def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
#         """
#         :param vocab_size: vocabulary size of the embedding layer
#         :param num_positions: max sequence length that will be fed to the model; should be 20
#         :param d_model: see TransformerLayer
#         :param d_internal: see TransformerLayer
#         :param num_classes: number of classes predicted at the output layer; should be 3
#         :param num_layers: number of TransformerLayers to use; can be whatever you want
#         """
#         super().__init__()
#         raise Exception("Implement me")

#     def forward(self, indices):
#         """

#         :param indices: list of input indices
#         :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
#         maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
#         """
#         raise Exception("Implement me")


# # Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# # of the same length, applying self-attention, the feedforward layer, etc.
# class TransformerLayer(nn.Module):
#     def __init__(self, d_model, d_internal):
#         """
#         :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
#         have to be the same size for the residual connection to work)
#         :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
#         should both be of this length.
#         """
#         super().__init__()
#         raise Exception("Implement me")

#     def forward(self, input_vecs):
#         raise Exception("Implement me")


# # Implementation of positional encoding that you can use in your network
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, num_positions: int=20, batched=False):
#         """
#         :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
#         added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
#         layer inputs/outputs)
#         :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
#         module will see
#         :param batched: True if you are using batching, False otherwise
#         """
#         super().__init__()
#         # Dict size
#         self.emb = nn.Embedding(num_positions, d_model)
#         self.batched = batched

#     def forward(self, x):
#         """
#         :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
#         :return: a tensor of the same size with positional embeddings added in
#         """
#         # Second-to-last dimension will always be sequence length
#         input_size = x.shape[-2]
#         indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
#         if self.batched:
#             # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
#             # gets added correctly across the batch
#             emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
#             return x + emb_unsq
#         else:
#             return x + self.emb(indices_to_embed)


# # This is a skeleton for train_classifier: you can implement this however you want
# def train_classifier(args, train, dev):
#     raise Exception("Not fully implemented yet")

#     # The following code DOES NOT WORK but can be a starting point for your implementation
#     # Some suggested snippets to use:
#     model = Transformer(...)
#     model.zero_grad()
#     model.train()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)

#     num_epochs = 10
#     for t in range(0, num_epochs):
#         loss_this_epoch = 0.0
#         random.seed(t)
#         # You can use batching if you'd like
#         ex_idxs = [i for i in range(0, len(train))]
#         random.shuffle(ex_idxs)
#         loss_fcn = nn.NLLLoss()
#         for ex_idx in ex_idxs:
#             loss = loss_fcn(...) # TODO: Run forward and compute loss
#             # model.zero_grad()
#             # loss.backward()
#             # optimizer.step()
#             loss_this_epoch += loss.item()
#     model.eval()
#     return model


# ####################################
# # DO NOT MODIFY IN YOUR SUBMISSION #
# ####################################
# def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
#     """
#     Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
#     :param model: your Transformer that returns log probabilities at each position in the input
#     :param dev_examples: the list of LetterCountingExample
#     :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
#     :param do_plot_attn: True if you want to write out plots for each example, false otherwise
#     :return:
#     """
#     num_correct = 0
#     num_total = 0
#     if len(dev_examples) > 100:
#         print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
#         do_print = False
#         do_plot_attn = False
#     for i in range(0, len(dev_examples)):
#         ex = dev_examples[i]
#         (log_probs, attn_maps) = model.forward(ex.input_tensor)
#         predictions = np.argmax(log_probs.detach().numpy(), axis=1)
#         if do_print:
#             print("INPUT %i: %s" % (i, ex.input))
#             print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
#             print("PRED %i: %s" % (i, repr(predictions)))
#         if do_plot_attn:
#             for j in range(0, len(attn_maps)):
#                 attn_map = attn_maps[j]
#                 fig, ax = plt.subplots()
#                 im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
#                 ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
#                 ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
#                 ax.xaxis.tick_top()
#                 # plt.show()
#                 plt.savefig("plots/%i_attns%i.png" % (i, j))
#         acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
#         num_correct += acc
#         num_total += len(predictions)
#     print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))



  # transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers,
                 use_positional_encoding=True, batched=False, task='BEFORE'):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer (size of keys/queries)
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use
        :param use_positional_encoding: whether to add positional embeddings
        :param batched: whether inputs will be batched (not required here)
        :param task: 'BEFORE' (default) => causal attention; 'BEFOREAFTER' => bidirectional attention
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.num_layers = num_layers
        self.task = task

        # Embedding for characters
        self.char_emb = nn.Embedding(vocab_size, d_model)

        # Positional encoding (provided in skeleton; uses embedding + addition)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(d_model, num_positions, batched=batched)

        # Stack of Transformer Layers
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])

        # Final classifier projecting each position's vector to num_classes
        self.classifier = nn.Linear(d_model, num_classes)

        # We'll return log probabilities
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _make_causal_mask(self, seq_len):
        """
        Creates a mask to prevent attention to future positions.
        We return a matrix of shape [seq_len, seq_len] where positions that are disallowed are -1e9,
        and allowed positions are 0. The mask will be added to raw attention scores before softmax.
        """
        # mask[i, j] should be 0 if j <= i, -inf if j > i
        mask = torch.zeros((seq_len, seq_len), dtype=torch.float32)
        # upper triangle (j > i) => -inf
        mask = mask.masked_fill(torch.triu(torch.ones_like(mask), diagonal=1).bool(), float(-1e9))
        return mask

    def forward(self, indices):
        """

        :param indices: list of input indices (torch.LongTensor of length seq_len) or batched [B, seq_len]
        :return: A tuple of the softmax log probabilities (should be a seq_len x num_classes matrix for
                 non-batched inputs or [B, seq_len, num_classes] for batched) and a list of the attention
                 maps you use in your layers (each is seq_len x seq_len or [B, seq_len, seq_len])
        """
        # Detect batched or not. In letter_counting driver, inputs are [seq_len] (no batch).
        is_batched = (indices.dim() == 2)
        if is_batched:
            batch_size, seq_len = indices.shape
        else:
            seq_len = indices.shape[0]
            batch_size = None

        x = self.char_emb(indices)  # if non-batched: [seq_len, d_model]; if batched: [B, seq_len, d_model]

        if self.use_positional_encoding:
            x = self.pos_enc(x)  # adds positional embeddings

        attn_maps_all_layers = []

        # Prepare attention mask if needed
        attn_mask = None
        if self.task != 'BEFOREAFTER':
            # causal mask for default "BEFORE" task
            cm = self._make_causal_mask(seq_len)  # [seq_len, seq_len]
            # If batched, we'll expand in layer when needed
            attn_mask = cm

        # Pass through stacked transformer layers
        out = x
        for layer in self.layers:
            # Our TransformerLayer.forward accepts attn_mask argument (or None)
            out, attn_map = layer.forward(out, attn_mask=attn_mask)
            attn_maps_all_layers.append(attn_map)

        # out is of same shape as input embedding: [seq_len, d_model] or [B, seq_len, d_model]
        logits = self.classifier(out)  # project to num_classes

        # return log probabilities (over last dim)
        log_probs = self.log_softmax(logits)

        return log_probs, attn_maps_all_layers


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
                        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Keys and queries
                           are both of this length. Values will be projected to d_model.
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal

        # Linear projections for queries, keys, values
        self.w_q = nn.Linear(d_model, d_internal, bias=False)
        self.w_k = nn.Linear(d_model, d_internal, bias=False)
        # Values project back to d_model
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Feedforward network (position-wise)
        # A small internal size is fine; we'll use d_internal * 2 for feedforward intermediate
        self.ff1 = nn.Linear(d_model, d_internal * 2)
        self.ff2 = nn.Linear(d_internal * 2, d_model)
        self.activation = nn.ReLU()

        # Softmax for attention
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_vecs, attn_mask=None):
        """
        :param input_vecs: Tensor of shape [seq_len, d_model] or [B, seq_len, d_model]
        :param attn_mask: None or tensor shape [seq_len, seq_len] with 0 for allowed and -inf for disallowed
        :return: (output_vecs, attn_probs)
                 output_vecs has same shape as input_vecs
                 attn_probs is attention probabilities (for plotting); shape [seq_len, seq_len] (non-batched) or
                 [B, seq_len, seq_len] for batched inputs
        """
        batched = (input_vecs.dim() == 3)
        if not batched:
            # Make it [1, seq_len, d_model] so we can handle both uniformly
            x = input_vecs.unsqueeze(0)  # [1, seq_len, d_model]
        else:
            x = input_vecs  # [B, seq_len, d_model]

        B, seq_len, _ = x.shape

        # Compute Q, K, V
        Q = self.w_q(x)  # [B, seq_len, d_internal]
        K = self.w_k(x)  # [B, seq_len, d_internal]
        V = self.w_v(x)  # [B, seq_len, d_model]

        # Compute attention scores: for each batch, scores = Q @ K^T -> [B, seq_len, seq_len]
        # Use matmul: Q [B, seq_len, d_internal] @ K.transpose(-2, -1) [B, d_internal, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, seq_len, seq_len]
        # Scale
        scores = scores / (np.sqrt(self.d_internal).astype(np.float32))

        # Apply mask if provided (attn_mask shape [seq_len, seq_len])
        if attn_mask is not None:
            # Expand mask to batch dimension
            # attn_mask has dtype float, with 0 allowed and -1e9 for disallowed
            if attn_mask.device != scores.device:
                attn_mask = attn_mask.to(scores.device)
            scores = scores + attn_mask.unsqueeze(0)  # [B, seq_len, seq_len]

        # Softmax to get attention probabilities over keys (last dim)
        attn_probs = self.softmax(scores)  # [B, seq_len, seq_len]

        # Multiply attention probs with V: attn_probs @ V -> [B, seq_len, d_model]
        attn_output = torch.matmul(attn_probs, V)

        # First residual connection
        res1 = x + attn_output  # [B, seq_len, d_model]

        # Feedforward (position-wise)
        ff_hidden = self.activation(self.ff1(res1))  # [B, seq_len, d_internal*2]
        ff_out = self.ff2(ff_hidden)  # [B, seq_len, d_model]

        # Second residual
        out = res1 + ff_out  # [B, seq_len, d_model]

        # Squeeze batch if input was not batched
        if not batched:
            out = out.squeeze(0)  # [seq_len, d_model]
            attn_probs_to_return = attn_probs.squeeze(0)  # [seq_len, seq_len]
        else:
            attn_probs_to_return = attn_probs  # [B, seq_len, seq_len]

        return out, attn_probs_to_return


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            if emb_unsq.device != x.device:
                emb_unsq = emb_unsq.to(x.device)
            return x + emb_unsq
        else:
            emb = self.emb(indices_to_embed)
            if emb.device != x.device:
                emb = emb.to(x.device)
            return x + emb


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    """
    Trains a Transformer classifier on the letter counting task.
    Supports both BEFORE and BEFOREAFTER modes from args.task.
    """

    # ==== Hyperparameters (fixed internally) ====
    d_model = 64          # embedding / hidden size
    d_internal = 32       # attention projection size
    num_layers = 1        # number of Transformer layers
    lr = 1e-3             # learning rate
    num_epochs = 1       # training epochs
    task = args.task      # BEFORE or BEFOREAFTER
    use_positional = True # always use positional encoding

    # ==== Infer vocabulary size and sequence length ====
    assert len(train) > 0, "Training data is empty!"
    sample_ex = train[0]
    seq_len = len(sample_ex.input_indexed)
    vocab_size = max(int(np.max(ex.input_indexed)) for ex in (train + dev)) + 1

    print(f"Training Transformer: task={task}, vocab={vocab_size}, seq_len={seq_len}")

    # ==== Initialize model ====
    model = Transformer(
        vocab_size=vocab_size,
        num_positions=seq_len,
        d_model=d_model,
        d_internal=d_internal,
        num_classes=3,
        num_layers=num_layers,
        use_positional_encoding=use_positional,
        batched=False,
        task=task
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fcn = nn.NLLLoss()

    # ==== Training loop ====
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        ex_indices = list(range(len(train)))
        random.shuffle(ex_indices)

        for idx in ex_indices:
            ex = train[idx]
            optimizer.zero_grad()

            log_probs, _ = model.forward(ex.input_tensor)  # [seq_len, num_classes]
            loss = loss_fcn(log_probs, ex.output_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}: Train loss = {total_loss:.4f} (time={elapsed:.2f}s)")
        model.eval()
        decode(model, dev, do_print=False, do_plot_attn=False)

    model.eval()
    return model



####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[ii] == ex.output[ii] for ii in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))

