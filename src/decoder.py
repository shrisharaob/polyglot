""" Decoder Module """
import torch
import torch.nn as nn
from transformer_module import TransformerModule
from self_attention import MultiAttention


class DecoderModule(nn.Module):
    """Documentation for Decoder

    """

    def __init__(self, embed_size, num_heads, dropout_ratio,
                 forward_expansion):
        super(DecoderModule, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout_ratio = dropout_ratio
        self.forward_expansion = forward_expansion
        # layers
        self.tgt_attention = MultiAttention(self.embed_size, self.num_heads)
        self.norm = nn.LayerNorm(self.embed_size)
        self.dec_transformer = TransformerModule(self.embed_size,
                                                 self.num_heads,
                                                 self.forward_expansion,
                                                 self.dropout_ratio)
        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, x, enc_keys, enc_context, src_mask, tgt_mask):
        """
        Keyword Arguments:
        src_context --
        mask        --
        """
        tgt_attention = self.tgt_attention(x, x, x, tgt_mask)
        queries = self.dropout(self.norm(x + tgt_attention))
        out = self.dec_transformer(enc_keys, enc_context, queries, src_mask)
        return out


class Decoder(nn.Module):
    """Documentation for Decoder

    """

    def __init__(self, tgt_vocab_size, embed_size, num_transformer_modules,
                 num_heads, dropout_ratio, forward_expansion, max_seq_len):
        super(Decoder, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size
        self.num_transformer_modules = num_transformer_modules
        self.num_heads = num_heads
        self.dropout_ratio = dropout_ratio
        self.forward_expansion = forward_expansion
        self.max_seq_len = max_seq_len
        # layers
        self.embedding = nn.Embedding(self.tgt_vocab_size, self.embed_size)
        self.position_embedding = nn.Embedding(self.max_seq_len,
                                               self.embed_size)

        self.dec_layers = nn.ModuleList([
            DecoderModule(self.embed_size, self.num_heads, self.dropout_ratio,
                          self.forward_expansion)
            for _ in range(self.num_transformer_modules)
        ])
        self.output_layer = nn.Linear(self.embed_size, self.tgt_vocab_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, tgt_sentences, tgt_mask, enc_context, src_mask):
        """

        """
        batch_size, seq_len = tgt_sentences.shape
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(positions)
        tgt_emb = self.embedding(tgt_sentences)
        x = self.dropout(pos_emb + tgt_emb)
        for layer in self.dec_layers:
            x = layer(x, enc_context, enc_context, src_mask, tgt_mask)
        out = self.output_layer(x)
        return out
