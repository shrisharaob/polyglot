""" Encoder Module """
import torch
import torch.nn as nn
from transformer_module import TransformerModule


class Encoder(nn.Module):
    """Documentation for Encoder

    """

    def __init__(self, src_vocab_size, embed_size, num_transformer_modules,
                 num_heads, dropout_ratio, forward_expansion, max_seq_len):
        super(Encoder, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.embed_size = embed_size
        self.num_transformer_modules = num_transformer_modules
        self.num_heads = num_heads
        self.dropout_ratio = dropout_ratio
        self.forward_expansion = forward_expansion
        self.max_seq_len = max_seq_len
        # layers
        self.embedding = nn.Embedding(self.src_vocab_size, self.embed_size)
        self.position_embedding = nn.Embedding(self.max_seq_len,
                                               self.embed_size)
        # transformer modules
        self.transforme_modules = nn.ModuleList(  #
            [
                TransformerModule(self.embed_size, self.num_heads,
                                  self.forward_expansion, self.dropout_ratio)
                for _ in range(self.num_transformer_modules)
            ])
        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, src_sentences, mask):
        # src_sentences dims: (batch_size, seq_len, embed_size)
        # mask dims: (seq_len, seq_len)
        batch_size, seq_len = src_sentences.shape
        emb = self.embedding(src_sentences)
        positions = torch.arange(seq_len).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(positions)
        out = self.dropout(emb + pos_emb)
        for layer in self.transforme_modules:
            out = layer(out, out, out, mask)
        return out
